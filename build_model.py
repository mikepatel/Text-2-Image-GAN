# Notes:
#   - TF v1.12
#   - CUDA v9.0
#   - cuDNN v7.4

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
from tensorflow.python.layers.convolutional import conv2d, conv2d_transpose
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
            return_sequences=False
        )(t)

    else:
        t = LSTM(
            units=128,
            return_sequences=False
        )(t)

    rnn_output = t

    m = Model(inputs=rnn_input, outputs=rnn_output)
    m.summary()
    return m


################################################################################
# GENERATOR
# using backend TF
def build_generator(noise, caption, reuse=False):
    with tf.variable_scope("generator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        #
        t_noise = noise
        t_txt = caption

        t_txt = dense(
            inputs=t_txt,
            units=128,
            activation=my_leaky_relu
        )

        t_input = tf.concat(
            values=[t_noise, t_txt],
            axis=1
        )

        t0 = dense(
            inputs=t_input,
            units=128*8*4*4
        )

        t0 = batch_normalization(
            inputs=t0
        )

        t0 = tf.reshape(
            tensor=t0,
            shape=[-1, 4, 4, 128*8]
        )

        t = conv2d(
            inputs=t0,
            filters=256,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="valid",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=256,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=128*8,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="same"
        )

        t = batch_normalization(
            inputs=t
        )

        t1 = tf.add(t0, t)

        t2 = conv2d_transpose(
            inputs=t1,
            filters=128*4,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding="same"
        )

        t2 = batch_normalization(
            inputs=t2
        )

        t = conv2d(
            inputs=t2,
            filters=128,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="valid",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=128,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=128*4,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="same"
        )

        t = batch_normalization(
            inputs=t
        )

        t3 = tf.add(t2, t)

        t4 = conv2d_transpose(
            inputs=t3,
            filters=128*2,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding="same",
            activation=my_leaky_relu
        )

        t4 = batch_normalization(
            inputs=t4
        )

        t5 = conv2d_transpose(
            inputs=t4,
            filters=128,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding="same",
            activation=my_leaky_relu
        )

        t5 = batch_normalization(
            inputs=t5
        )

        t_output = conv2d_transpose(
            inputs=t5,
            filters=3,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding="same",
            activation=tf.tanh
        )

        print("\nGenerator Output Shape: {}".format(t_output.shape))
        return t_output


################################################################################
# DISCRIMINATOR
# using backend TF
def build_discriminator(image, caption, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        #
        t0 = image

        t0 = conv2d(
            inputs=t0,
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="same",
            activation=my_leaky_relu
        )

        t1 = conv2d(
            inputs=t0,
            filters=128,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="same",
            activation=my_leaky_relu
        )

        t1 = batch_normalization(
            inputs=t1
        )

        t2 = conv2d(
            inputs=t1,
            filters=256,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="same",
            activation=my_leaky_relu
        )

        t2 = batch_normalization(
            inputs=t2
        )

        t3 = conv2d(
            inputs=t2,
            filters=512,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="same",
            activation=None
        )

        t3 = batch_normalization(
            inputs=t3
        )

        t = conv2d(
            inputs=t3,
            filters=128,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="valid",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=128,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="same",
            activation=my_leaky_relu
        )

        t = batch_normalization(
            inputs=t
        )

        t = conv2d(
            inputs=t,
            filters=512,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="same"
        )

        t = batch_normalization(
            inputs=t
        )

        t4 = tf.add(t3, t)

        #
        t_txt = dense(
            inputs=caption,
            units=128,
            activation=my_leaky_relu
        )

        t_txt = tf.expand_dims(
            input=t_txt,
            axis=1
        )

        t_txt = tf.expand_dims(
            input=t_txt,
            axis=1
        )

        t_txt = tf.tile(
            input=t_txt,
            multiples=[1, 4, 4, 1]
        )

        t_concat = tf.concat(
            values=[t4, t_txt],
            axis=3
        )

        t4 = conv2d(
            inputs=t_concat,
            filters=512,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="valid",
            activation=my_leaky_relu
        )

        t4 = batch_normalization(
            inputs=t4
        )

        t_output = conv2d(
            inputs=t4,
            filters=1,
            kernel_size=[4, 4],
            strides=[4, 4],
            padding="valid"
        )

        print("\nDiscriminator Output Shape: {}".format(t_output.shape))
        return t_output
