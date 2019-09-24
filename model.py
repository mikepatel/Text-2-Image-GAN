"""
Michael Patel
August 2018

Python 3.6.5
TensorFlow 1.12.0

Project description:
    To synthesize images from text-based captions as a form of reverse image captioning.

File description:
    To build model definitions for CNN, RNN, Generator, Discriminator

Dataset: Oxford-102 flowers

Notes:
    - using Keras to build CNN and RNN
    - using backend TensorFlow to build Generator and Discriminator

"""
################################################################################
# Imports
import tensorflow as tf
import tensorflow.python.keras as k

from parameters import IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, LEAKY_ALPHA


################################################################################
# Leaky ReLU
def my_leaky_relu(tensor):
    return tf.nn.leaky_relu(tensor, alpha=LEAKY_ALPHA)


################################################################################
# CNN for text embedding
def build_cnn():
    # Input
    cnn_inputs = k.Input(
        shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS)
    )

    t = cnn_inputs

    # Convolutional layer #1
    t = k.layers.Conv2D(
        filters=64,
        kernel_size=[4, 4],
        strides=2,
        input_shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS),
        padding="same",
        activation=my_leaky_relu
    )(t)

    # Convolutional layer #2
    t = k.layers.Conv2D(
        filters=128,
        kernel_size=[4, 4],
        strides=2,
        padding="same",
        activation=my_leaky_relu
    )(t)

    # Batchnormalization layer #1
    t = k.layers.BatchNormalization()(t)

    # Convolutional layer #3
    t = k.layers.Conv2D(
        filters=256,
        kernel_size=[4, 4],
        strides=2,
        activation=my_leaky_relu
    )(t)

    # Batchnormalization layer #2
    t = k.layers.BatchNormalization()(t)

    # Convolutional layer #4
    t = k.layers.Conv2D(
        filters=512,
        kernel_size=[4, 4],
        strides=2,
        activation=my_leaky_relu
    )(t)

    # Batchnormalization layer #3
    t = k.layers.BatchNormalization()(t)

    # Flatten layer
    t = k.layers.Flatten()(t)

    # Dense Output layer
    t = k.layers.Dense(
        units=128,
        activation=None  # linear activation
    )(t)

    cnn_outputs = t

    m = k.Model(inputs=cnn_inputs, outputs=cnn_outputs)
    m.summary()
    return m


################################################################################
