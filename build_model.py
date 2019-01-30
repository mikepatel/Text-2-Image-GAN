# TF v1.12

# TF backend
# Keras

# Contains functions to build the following models:
#   - CNN (for text classification / text embedding)
#   - RNN
#   - Generator
#   - Discriminator


################################################################################
# IMPORTs
import tensorflow as tf

# backend TF
#from tensorflow.python.keras.backend import conv2d, batch_normalization, flatten
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.core import dense, flatten
from tensorflow.python.layers.normalization import batch_normalization

# Keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Embedding, CuDNNLSTM, LSTM, \
    Conv2D, Dense, Flatten, BatchNormalization


################################################################################
# HYPERPARAMETERS and DESIGN CHOICES
LEAKY_ALPHA = 0.2

# image dimensions: 64x64x3
IMAGE_ROWS = 64
IMAGE_COLS = 64
IMAGE_CHANNELS = 3


################################################################################
# Leaky ReLU
def my_leaky_relu(tensor):
    return tf.nn.leaky_relu(tensor, alpha=LEAKY_ALPHA)


################################################################################
# CNN for text embedding
# using Keras
def build_cnn():
    cnn_inputs = Input(
        shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS),
    )

    t = cnn_inputs

    t = Conv2D(
        filters=64,
        kernel_size=[4, 4],
        strides=2,
        input_shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS),
        padding="same",
        activation=my_leaky_relu
    )(t)

    t = Conv2D(
        filters=128,
        kernel_size=[4, 4],
        strides=2,
        padding="same",
        activation=my_leaky_relu
    )(t)

    t = BatchNormalization()(t)

    t = Conv2D(
        filters=256,
        kernel_size=[4, 4],
        strides=2,
        activation=my_leaky_relu
    )(t)

    t = BatchNormalization()(t)

    t = Conv2D(
        filters=512,
        kernel_size=[4, 4],
        strides=2,
        activation=my_leaky_relu
    )(t)

    t = BatchNormalization()(t)

    t = Flatten()(t)

    t = Dense(
        units=128,
        activation=None  # linear activation
    )(t)

    cnn_out = t

    m = Model(inputs=cnn_inputs, outputs=cnn_out)
    m.summary()
    return m


################################################################################
# RNN for text embedding
# using Keras
def build_rnn():
    rnn_input = Input(
        shape=(None, )
    )

    t = Embedding(
        input_dim=8000,  # vocab size
        output_dim=256  # word embedding size
    )(rnn_input)

    # GPU check
    if tf.test.is_gpu_available():
        t = CuDNNLSTM(
            units=128,
            return_sequences=True
        )(t)

    else:
        t = LSTM(
            units=128,
            return_sequences=True
        )(t)

    rnn_output = t

    m = Model(inputs=rnn_input, outputs=rnn_output)
    m.summary()
    return m


################################################################################
# GENERATOR
# using backend TF


################################################################################
# DISCRIMINATOR
# using backend TF
