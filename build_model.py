# Contains functions to build the following models:
#   - CNN
#   - RNN
#   - Generator
#   - Discriminator


################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.layers import conv2d, batch_normalization, flatten, dense


################################################################################
LEAKY_ALPHA = 0.2

# image dimensions: 64x64x3
IMAGE_ROWS = 64
IMAGE_COLUMNS = 64
IMAGE_CHANNELS = 3


################################################################################
def my_leaky_relu(tensor):
    return tf.nn.leaky_relu(tensor, alpha=LEAKY_ALPHA)


################################################################################
# CNN for text embedding
def build_cnn(image, reuse=False):
    with tf.variable_scope("cnn") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        #
        cnn_input = image

        t = conv2d(
            inputs=cnn_input,
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
            filters=132,
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
def build_rnn(reuse=False):
    with tf.variable_scope("rnn") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        #
