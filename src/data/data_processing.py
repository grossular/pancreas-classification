"""Data processsing helpers"""
import torch
from torchvision import datasets, transforms
import numpy as np


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for
       imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))
                            ) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(
            self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def load_split_train_test(datadir, batch_size, valid_size=.2, crop_size=224,
                          sampler='SubsetRandom', crop_strategy='ten_crop'):
    """Load data from datadir into training set and validation set
    Arguments:
        datadir (string): Path to data
        batch_size (int): Number of images in each batch
        valid_size (float): Proportion of images to use in validation set
        crop_size (int): Size images will be resized to (square)
        sampler (string): Sampling strategy
        crop_strategy (string): Augmentation crop to use. Only supports 'ten_crop" currently
    """
    if crop_strategy is 'ten_crop':
        train_transforms = transforms.Compose([transforms.Resize(crop_size*2),
                                               transforms.TenCrop(crop_size),
                                               transforms.Lambda(lambda crops: torch.stack([
                                                   transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize([0.485, 0.456, 0.406], [
                                                                           0.229, 0.224, 0.225])])
                                                   (crop) for crop in crops])),
                                               ])

        test_transforms = train_transforms
    else:
        train_transforms = transforms.Compose([transforms.Resize(224),
                                               transforms.ToTensor(),
                                               ])

        test_transforms = train_transforms

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    # Create indices for train/test split
    num_train = len(train_data)
    num_test = len(test_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    print(
        f'Loaded {num_train-split} training samples and {split} validation samples')
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    # Setup sampling strategy
    if sampler is 'SubsetRandom':
        from torch.utils.data.sampler import SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
    elif sampler is 'RandomSamplerRaw':
        from torch.utils.data.sampler import RandomSampler
        train_sampler = RandomSampler(train_idx)
        test_sampler = RandomSampler(test_idx)
    elif sampler is 'RandomSamplerRep':
        from torch.utils.data.sampler import RandomSampler
        train_sampler = RandomSampler(
            train_idx, replacement=True, num_samples=num_train)
        test_sampler = RandomSampler(
            test_idx, replacement=True, num_samples=num_test)
    elif sampler is 'Balanced':
        train_sampler = ImbalancedDatasetSampler(train_data, train_idx)
        test_sampler = ImbalancedDatasetSampler(test_data, test_idx)
    else:
        train_sampler = None
        test_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(
        test_data, sampler=test_sampler, batch_size=batch_size)

    return trainloader, testloader
