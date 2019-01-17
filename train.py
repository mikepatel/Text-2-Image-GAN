# Notes:
#   - Using both backend TF and Keras
#   - TF v1.12


############################################################
# IMPORTS
import tensorflow as tf
import keras.backend as k

import os
import sys
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from build_model import build_cnn, build_rnn


############################################################
# HYPERPARAMETERS and DESIGN CHOICES
BATCH_SIZE = 64
NUM_EPOCHS = 1

# image dimensions: 64x64x3
IMAGE_ROWS = 64
IMAGE_COLS = 64
IMAGE_CHANNELS = 3

Z_DIM = 512


############################################################
# SETUP for Callbacks and Generated Images

# TF version
print("\nTF version: {}".format(tf.__version__))

# create folder to save checkpoints
save_folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


############################################################
# LOAD DATA
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("_image_train.pickle", 'rb') as f:
    _, images_train = pickle.load(f)
with open("_image_test.pickle", 'rb') as f:
    _, images_test = pickle.load(f)
with open("_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)


############################################################
# PREPROCESS DATA
images_train = np.array(images_train)
images_test = np.array(images_test)


############################################################
# RNN
# CNN for text embedding is built using backend TF
# LSTM for text embedding is built using Keras
def train_rnn():
    print("\nTraining RNN...")

    tf.reset_default_graph()
    k.set_learning_phase(1)  # 1 = train, 0 = test

    ##########
    # Placeholders
    real_image_pl = tf.placeholder(dtype=tf.float32,
                                   shape=[None, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS],
                                   name="real_image_pl")

    wrong_image_pl = tf.placeholder(dtype=tf.float32,
                                    shape=[None, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS],
                                    name="wrong_image_pl")

    real_caption_pl = tf.placeholder(dtype=tf.float32,
                                     shape=[None, None],
                                     name="real_caption_pl")

    wrong_caption_pl = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None],
                                      name="wrong_caption_pl")

    ##########
    # Instantiate models
    # using backend TF to build model
    cnn_real_image = build_cnn(real_image_pl, reuse=False)
    cnn_wrong_image = build_cnn(wrong_image_pl, reuse=True)

    # using Keras to build model
    rnn = build_rnn()
    rnn_real_caption = rnn(real_caption_pl)
    rnn_wrong_caption = rnn(wrong_caption_pl)

    ##########
    # Loss function
    def cosine_similarity(a, b):
        """
        https://en.wikipedia.org/wiki/Cosine_similarity#Definition

        :param a: tensor of [batch_size, n_feature]
        :param b: tensor of [batch_size, n_feature]
        :return: tensor of [batch_size, ]
        """

        cs = (tf.reduce_sum(tf.multiply(a, b), axis=1)) / \
             (
                     tf.sqrt(tf.reduce_sum(tf.multiply(a, a), axis=1)) *
                     tf.sqrt(tf.reduce_sum(tf.multiply(b, b), axis=1))
             )
        return cs

    alpha = 0.2
    rnn_loss = tf.reduce_mean(tf.maximum(0.,
                                         alpha -
                                         cosine_similarity(cnn_real_image, rnn_real_caption) +
                                         cosine_similarity(cnn_real_image, rnn_wrong_caption))) + \
               tf.reduce_mean(tf.maximum(0.,
                                         alpha -
                                         cosine_similarity(cnn_real_image, rnn_real_caption) +
                                         cosine_similarity(cnn_wrong_image, rnn_real_caption)))

    ##########
    # Optimizer
    # .minimize() takes care of both computing the gradients and applying them to variables
    # https://www.tensorflow.org/api_docs/python/tf/train/Optimizer#processing_gradients_before_applying_them
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5)

    # compute gradients
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(rnn_loss,
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cnn") + rnn.trainable_weights
                     ), clip_norm=10)

    # apply gradients
    rnn_optimizer = optimizer.apply_gradients(zip(grads,
                                                  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cnn") +
                                                  rnn.trainable_weights))

    ##########
    # Session initialization and TensorBoard Setup
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.summary.scalar(name="RNN Loss", tensor=rnn_loss)
    tb = tf.summary.merge_all()
    tb_writer = tf.summary.FileWriter(logdir=save_folder, graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    ##########
    # training loop
    for epoch in range(NUM_EPOCHS+1):
        if epoch % 100 == 0:
            print("\nEpoch: {}".format(epoch))
            print("RNN Loss {}".format())

    rnn.save_weights("rnn_weights.h5")

############################################################
# GAN
def train_gan():
    print("\nTraining GAN...")


############################################################
# MAIN
if __name__ == "__main__":
    if "--rnn" in sys.argv:
        train_rnn()

    if "--gan" in sys.argv:
        train_gan()
