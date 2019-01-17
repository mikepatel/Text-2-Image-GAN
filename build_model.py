# TF v1.12

# Contains functions to build the following models:
#   - CNN (for text classification / text embedding)
#   - RNN
#   - Generator
#   - Discriminator


################################################################################
# IMPORTs
import tensorflow as tf

# backend TF
from tensorflow.python.keras.backend import conv2d, batch_normalization, flatten
from tensorflow.python.layers.core import dense

# Keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Embedding, CuDNNLSTM, LSTM


################################################################################
# HYPERPARAMETERS and DESIGN CHOICES
LEAKY_ALPHA = 0.2

'''
# image dimensions: 64x64x3
IMAGE_ROWS = 64
IMAGE_COLS = 64
IMAGE_CHANNELS = 3
'''


################################################################################
# Leaky ReLU
def my_leaky_relu(tensor):
    return tf.nn.leaky_relu(tensor, alpha=LEAKY_ALPHA)


################################################################################
# CNN for text embedding
# using backend TF
def build_cnn(image, reuse=False):
    with tf.variable_scope("cnn") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        #
        t = image  # input

        t = conv2d(
            inputs=t,
            filters=64,
            kernel_size=[4, 4],
            strides=2,
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=128,
            kernel_size=[4, 4],
            strides=2,
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=256,
            kernel_size=[4, 4],
            strides=2,
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=512,
            kernel_size=[4, 4],
            strides=2,
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = flatten(
            inputs=t
        )

        t = dense(
            inputs=t,
            units=128,
            activation=None  # linear activation
        )

        cnn_output = t
        return cnn_output


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
