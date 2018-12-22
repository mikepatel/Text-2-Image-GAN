# Contains functions to build the following models:
#   - CNN
#   - RNN
#   - Generator
#   - Discriminator


################################################################################
# IMPORTs
import tensorflow as tf
import tensorflow.layers


################################################################################

# image dimensions: 64x64x3
IMAGE_ROWS = 64
IMAGE_COLUMNS = 64
IMAGE_CHANNELS = 3


################################################################################
def build_cnn(image, reuse=False):
    
