"""Define and train/test pre-trained models"""
import sys
import torch
from torch import optim, nn
from tqdm import tqdm
from torchvision import models
import numpy as np
from sklearn.metrics import confusion_matrix


def set_parameter_requires_grad(model):
    """Freeze model Gradiends
    Arguments:
        model (object): pytorch model
    """
    for param in model.parameters():
        param.requires_grad = False

def get_model(model_name='alexnet', num_classes=2, lr=0.05):
    """Get pre-trained model with additional layer suitable for num_classes
    Arguments:
        model_name (string): model to load
        num_classes: (int): number of unique labels/classes
        lr (float): learning rate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    elif model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(pretrained=True)
        set_parameter_requires_grad(model)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        set_parameter_requires_grad(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)

    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    return model, device, optimizer, criterion

def train_model(model, device, optimizer, criterion, trainloader, testloader, epochs,
                inception=False, model_name='test'):
    """Train the model
    Arguments:
        model (object): pytorch model
        device (string): 'cpu' or 'gpu'
        criterion (object): pytorch loss
        trainloader (object): pytorch dataloader of training samples
        testloader (object): pytorch dataloader of testing samples
        epochs (int): number of epochs to train
        inception (bool): If model is of type 'inception'
        model_name (string): Name of running model
    """
    steps = 0
    running_loss = 0
    train_losses, test_losses, test_accuracy, = [], [], []
    print(f'Training {model_name}')
    sys.stdout.flush()
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            bs, ncrops, c, h, w = inputs.size()
            # For inception we must handle the auxilary net
            if inception:
                logps1, logps2 = model.forward(inputs.view(-1, c, h, w))
                logps1 = logps1.view(bs, ncrops, -1).mean(1)
                loss1 = criterion(logps1, labels)
                logps2 = logps2.view(bs, ncrops, -1).mean(1)
                loss2 = criterion(logps2, labels)
                loss = loss1 + loss2
            else:
                logps = model.forward(inputs.view(-1, c, h, w))
                logps = logps.view(bs, ncrops, -1).mean(1)
                loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for valid_inputs, valid_labels in testloader:
                    valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                    bs, ncrops, c, h, w = valid_inputs.size()
                    logps = model.forward(valid_inputs.view(-1, c, h, w))
                    logps = logps.view(bs, ncrops, -1).mean(1)
                    batch_loss = criterion(logps, valid_labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == valid_labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            model.train()
        test_accuracy.append(accuracy/len(testloader))
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

    return model, train_losses, test_losses, test_accuracy

def test_model(model, device, testloader, model_name='test'):
    """Test the model
    Arguments:
        model (object): pytorch model
        device (string): 'cpu' or 'gpu'
        testloader (object): pytorch dataloader of training samples
    """
    num_classes = len(testloader.dataset.classes)
    correct = 0
    total = 0
    y_true, y_score = [], []
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int16)
    model.eval()
    with torch.no_grad():
        print(f'Testing: {model_name}')
        sys.stdout.flush()
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            bs, ncrops, c, h, w = inputs.size()
            output = model(inputs.view(-1, c, h, w))
            output = output.view(bs, ncrops, -1).mean(1)
            pred = torch.argmax(output, 1)
            total += labels.size(0)
            y_true.append(labels.cpu().numpy())
            y_score.append(output.cpu().numpy())
            correct += (pred == labels).sum().item()
            conf_matrix = np.add(conf_matrix, confusion_matrix(labels.cpu(), pred.cpu()))
        try:
            test_acc = 100. * correct / total
        except ZeroDivisionError:
            test_acc = 0
    model.train()

    return conf_matrix, test_acc, y_true, y_score
