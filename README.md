# pancreas-classification
Image classification of retrieval pancreases with the aim of informing transplant viability

## Training a new model
```
usage: train_model.py [-h] -d DATA_DIR -m MODEL [-s SAMPLE_METHOD]
                      [-b BATCH_SIZE] [-lr LEARNING_RATE] [-v VALID_SIZE]
                      [-e EPOCHS] [-f FIG_DIR] [-o OUTPUT_DIR] [-md MODEL_DIR]

Use transfer learning on pre-trained models to make new image classifier

required arguments:
  -d DATA_DIR, --data_dir DATA_DIR
                        Path to data dir. Path must contain dirs of images
                        named for each class
  -m MODEL, --model MODEL
                        Pre-trained model to use. Options: alexnet, vgg19,
                        resnet152, squeezenet1_1, densenet201, inception

optional arguments:
  -h, --help            show this help message and exit
  -s SAMPLE_METHOD, --sample_method SAMPLE_METHOD
                        Data sampling method to use. Default: SubsetRandom
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of images per batch. Default: 20
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Model learning rate. Default: 0.01
  -v VALID_SIZE, --valid_size VALID_SIZE
                        Proportion of images to use for validation. Default:
                        0.2
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train the model. Default: 50
  -f FIG_DIR, --fig_dir FIG_DIR
                        Directory to save generated figures. Default:
                        data/results/training/figures
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory to save generated output. Default:
                        data/results/training/csv
  -md MODEL_DIR, --model_dir MODEL_DIR
                        Directory to save the trained model. Default:
                        data/models
```
## Testing a trained model
```
usage: test_model.py [-h] -d DATA_DIR -m MODEL_PATH [-b BATCH_SIZE]
                     [-v VALID_SIZE] [-f FIG_DIR] [-o OUTPUT_DIR]

Use transfer learning on pre-trained models to make new image classifier

required arguments:
  -d DATA_DIR, --data_dir DATA_DIR
                        Path to data dir. Path must contain dirs of images
                        named for each class
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to model to test

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of images per batch. Default: 20
  -v VALID_SIZE, --valid_size VALID_SIZE
                        Proportion of images to use for validation. Default:
                        0.2
  -f FIG_DIR, --fig_dir FIG_DIR
                        Directory to save generated figures. Default:
                        data/results/testing/figures
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory to save generated output. Default:
                        data/results/testing/csv
```