"""Use transfer learning on pretrained model. Save model and outputs"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from data.data_processing import load_split_train_test
from data.plots import plot_roc, class_bar_plot, confusion_matrix_plot
from models.pretrained import get_model, test_model, train_model

PARSER = argparse.ArgumentParser(
        description='Use transfer learning on pre-trained models to make new image classifier')
OPTIONAL = PARSER._action_groups.pop()
REQUIRED = PARSER.add_argument_group('required arguments')
REQUIRED.add_argument('-d', '--data_dir',
                      help='Path to data dir. Path must contain dirs of images named as each class',
                      required=True)
REQUIRED.add_argument('-m', '--model',
                      help='Pre-trained model to use. Options: alexnet, vgg19, \
                      resnet152, squeezenet1_1, densenet201, inception', required=True)
OPTIONAL.add_argument('-s', '--sample_method',
                      help='Data sampling method to use. Default: SubsetRandom',
                      default='SubsetRandom')
OPTIONAL.add_argument('-b', '--batch_size',
                      help='Number of images per batch. Default: 20',
                      default=20)
OPTIONAL.add_argument('-lr', '--learning_rate',
                      help='Model learning rate. Default: 0.01',
                      default=0.01)
OPTIONAL.add_argument('-v', '--valid_size',
                      help='Proportion of images to use for validation. Default: 0.2',
                      default=0.2)
OPTIONAL.add_argument('-e', '--epochs', help='Number of epochs to train the model. \
                      Default: 50',
                      default=50)
OPTIONAL.add_argument('-f', '--fig_dir',
                      help='Directory to save generated figures. \
                      Default: data/results/training/figures',
                      default='data/results/training/figures')
OPTIONAL.add_argument('-o', '--output_dir',
                      help='Directory to save generated output. \
                      Default: data/results/training/csv',
                      default='data/results/training/csv')
OPTIONAL.add_argument('-md', '--model_dir',
                      help='Directory to save the trained model. \
                      Default: data/models',
                      default='data/models')
PARSER._action_groups.append(OPTIONAL)
args = PARSER.parse_args()

RUN_NAME = f"{args.data_dir.replace('/', '-')}-{args.model}"
EPOCHS = args.epochs
if args.model == 'inception':
    CROP_SIZE = 299
    INCEPTION = True
else:
    CROP_SIZE = 224
    INCEPTION = False

# Ensure output directories exist
for folder in [args.fig_dir, args.output_dir, args.model_dir]:
    try:
        os.makedirs(folder)
    except FileExistsError:
        # directory already exists
        pass

# Load data
TRAIN_LOADER, TEST_LOADER = load_split_train_test(
    args.data_dir, args.batch_size, args.valid_size, CROP_SIZE, args.sample_method)
NUM_CLASSES = len(TRAIN_LOADER.dataset.classes)

# Plot class counts
class_bar_plot(TRAIN_LOADER, args.fig_dir, RUN_NAME)

# Load the required model
MODEL, DEVICE, OPTIMIZER, CRITERION = get_model(
    model_name=args.model, num_classes=NUM_CLASSES, lr=args.learning_rate)

# Train, test and save the model
MODEL, TRAIN_LOSSES, TEST_LOSSES, TEST_ACCURACY = train_model(
    MODEL, DEVICE, OPTIMIZER, CRITERION, TRAIN_LOADER, TEST_LOADER, EPOCHS,
    inception=INCEPTION, model_name=args.model)
CONF_MATRIX, TEST_ACC, Y_TRUE, Y_SCORE = test_model(MODEL, DEVICE, TEST_LOADER, args.model)
torch.save(MODEL, f'{args.model_dir}{os.sep}{RUN_NAME}.model')

# Save output
confusion_matrix_plot(TEST_LOADER, CONF_MATRIX, args.fig_dir, RUN_NAME)
np.savetxt(f'{args.output_dir}{os.sep}{RUN_NAME}-conf_matrix.csv',
           CONF_MATRIX, fmt='%1.8f', delimiter=',', newline='\n')
with open(f'{args.output_dir}{os.sep}{RUN_NAME}-final_accuracy.csv', 'w') as f:
    f.write(str(TEST_ACC))
np.savetxt(f'{args.output_dir}{os.sep}{RUN_NAME}-validation_accuracy.csv',
           TEST_ACCURACY, fmt='%1.8f', delimiter=',', newline='\n')
np.savetxt(f'{args.output_dir}{os.sep}{RUN_NAME}-training_loss.csv',
           TRAIN_LOSSES, fmt='%1.8f', delimiter=',', newline='\n')
np.savetxt(f'{args.output_dir}{os.sep}{RUN_NAME}-validation_loss.csv',
           TEST_LOSSES, fmt='%1.8f', delimiter=',', newline='\n')
plt.plot(TEST_ACCURACY, label='Validation Accuracy')
plt.plot(TRAIN_LOSSES, label='Training loss')
plt.plot(TEST_LOSSES, label='Validation loss')
plt.legend(frameon=False)
plt.savefig(f'{args.fig_dir}{os.sep}{RUN_NAME}_acc_loss.pdf')
plt.clf()
plot_roc(Y_TRUE, Y_SCORE, NUM_CLASSES, args.output_dir, args.fig_dir, RUN_NAME)
