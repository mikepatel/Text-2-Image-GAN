"""
Michael Patel
August 2018

Python 3.6.5
TensorFlow 1.12.0

Project description:
    To synthesize images from text-based captions as a form of reverse image captioning.

File description:
    To hold constants and model hyperparameters

Dataset: Oxford-102 flowers

"""
################################################################################
# image dimensions: 64x64x3
IMAGE_ROWS = 64
IMAGE_COLS = 64
IMAGE_CHANNELS = 3

# Training
NUM_EPOCHS = 10000
BATCH_SIZE = 64
G_LEARNING_RATE = 0.0002
D_LEARNING_RATE = 0.0002

# Model
Z_DIM = 512
LEAKY_ALPHA = 0.2
