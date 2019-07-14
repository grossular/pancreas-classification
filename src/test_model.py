"""Test trained model"""
import os
import argparse
import torch
import numpy as np
from models.pretrained import test_model
from data.data_processing import load_split_train_test
from data.plots import plot_roc, class_bar_plot, confusion_matrix_plot

PARSER = argparse.ArgumentParser(
        description='Use transfer learning on pre-trained models to make new image classifier')
OPTIONAL = PARSER._action_groups.pop()
REQUIRED = PARSER.add_argument_group('required arguments')
REQUIRED.add_argument('-d', '--data_dir',
                      help='Path to data dir. Path must contain dirs of images named as each class',
                      required=True)
REQUIRED.add_argument('-m', '--model_path',
                      help='Path to model to test', required=True)
OPTIONAL.add_argument('-b', '--batch_size',
                      help='Number of images per batch. Default: 20',
                      default=20)
OPTIONAL.add_argument('-f', '--fig_dir',
                      help='Directory to save generated figures. \
                      Default: data/results/testing/figures',
                      default='data/results/testing/figures')
OPTIONAL.add_argument('-o', '--output_dir',
                      help='Directory to save generated output. \
                      Default: data/results/testing/csv',
                      default='data/results/testing/csv')
PARSER._action_groups.append(OPTIONAL)
args = PARSER.parse_args()

# Ensure output directories exist
for folder in [args.fig_dir, args.output_dir]:
    try:
        os.makedirs(folder)
    except FileExistsError:
        # directory already exists
        pass


RUN_NAME = args.model_path.split('.')[0].replace('/', '-')
MODEL_NAME = args.model_path.split('-')[-1].split('.')[0]

if 'inception' in args.model_path:
    CROP_SIZE = 299
    INCEPTION = True
else:
    CROP_SIZE = 224
    INCEPTION = False

EVAL_LOADER, _ = load_split_train_test(args.data_dir, args.batch_size,
                                       0, CROP_SIZE, None)

MODEL = torch.load(args.model_path)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(EVAL_LOADER.dataset.classes)
CONF_MATRIX, TEST_ACC, Y_TRUE, Y_SCORE = test_model(MODEL, DEVICE, EVAL_LOADER, MODEL_NAME)

# Plot class counts
class_bar_plot(EVAL_LOADER, args.fig_dir, RUN_NAME)

# Save output
confusion_matrix_plot(EVAL_LOADER, CONF_MATRIX, args.fig_dir, RUN_NAME)
np.savetxt(f'{args.output_dir}{os.sep}{RUN_NAME}-conf_matrix.csv',
           CONF_MATRIX, fmt='%1.8f', delimiter=',', newline='\n')
with open(f'{args.output_dir}{os.sep}{RUN_NAME}-faccuracy.csv', 'w') as f:
    f.write(str(TEST_ACC))

plot_roc(Y_TRUE, Y_SCORE, NUM_CLASSES, args.output_dir, args.fig_dir, RUN_NAME, )
