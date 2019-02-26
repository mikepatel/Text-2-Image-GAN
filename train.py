# Notes:
#   - Based on the work of: https://arxiv.org/pdf/1605.05396.pdf
#   - Using both backend TF and Keras
#   - TF v1.12
#   - CUDA v9.0
#   - cuDNN v7.4

#   - LSTM => return_sequences boolean


############################################################
# IMPORTS
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import os
import sys
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from build_model import build_cnn, build_rnn, build_discriminator, build_generator
from utils import *


############################################################
# HYPERPARAMETERS and DESIGN CHOICES
BATCH_SIZE = 64
NUM_EPOCHS = 10000
LEARNING_RATE = 0.0002

# image dimensions: 64x64x3
IMAGE_ROWS = 64
IMAGE_COLS = 64
IMAGE_CHANNELS = 3

Z_DIM = 512


############################################################
def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.

    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [np.random.randint(min, max) for p in range(0, number)]


############################################################
# SETUP for Callbacks and Generated Images

# TF version
print("\nTF version: {}".format(tf.__version__))


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
# CNN for text embedding is built using Keras
# LSTM for text embedding is built using Keras
def train_rnn():
    print("\nTraining RNN...")

    # create folder to save checkpoints
    save_folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(1)  # 1=train, 0=test
    #k.set_learning_phase(1)  # 1 = train, 0 = test

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
    # using Keras to build model
    cnn = build_cnn()
    cnn_real_image = cnn(real_image_pl)
    cnn_wrong_image = cnn(wrong_image_pl)

    # using Keras to build model
    rnn = build_rnn()
    rnn_real_caption = rnn(real_caption_pl)
    rnn_wrong_caption = rnn(wrong_caption_pl)

    print("\nCNN embed output shape: {}".format(cnn_real_image.shape))
    print("RNN embed output shape: {}\n".format(rnn_real_caption.shape))

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

    #
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
                     cnn.trainable_weights + rnn.trainable_weights
                     ), clip_norm=10.0)

    # apply gradients
    rnn_optimizer = optimizer.apply_gradients(zip(grads,
                                                  cnn.trainable_weights + rnn.trainable_weights))

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
        # right captions
        cap_idx = get_random_int(min=0, max=n_captions_train - 1, number=BATCH_SIZE)
        real_caps = captions_ids_train[cap_idx]
        real_caps = pad_sequences(real_caps, maxlen=128, padding="post")  # pad captions to fixed length

        # right images
        img_idx = np.floor(np.asarray(cap_idx).astype("float") / n_captions_per_image).astype("int")
        real_images = images_train[img_idx]

        # wrong caption
        cap_idx = get_random_int(min=0, max=n_captions_train - 1, number=BATCH_SIZE)
        wrong_caps = captions_ids_train[cap_idx]
        wrong_caps = pad_sequences(wrong_caps, maxlen=128, padding="post")

        # wrong images
        img_idx = get_random_int(min=0, max=n_images_train - 1, number=BATCH_SIZE)
        wrong_images = images_train[img_idx]

        # preprocessing on images
        real_images = threading_data(real_images, prepro_img, mode="train")
        wrong_images = threading_data(wrong_images, prepro_img, mode="train")

        rnn_error, _ = sess.run(
            [rnn_loss, rnn_optimizer],
            feed_dict={
                real_caption_pl: real_caps,
                real_image_pl: real_images,
                wrong_caption_pl: wrong_caps,
                wrong_image_pl: wrong_images
            }
        )

        if epoch % 10 == 0:
            print("\nEpoch: {}".format(epoch))
            print("RNN Loss {}\n".format(rnn_error))

            # TensorBoard
            summary = sess.run(
                tb,
                feed_dict={
                    real_caption_pl: real_caps,
                    real_image_pl: real_images,
                    wrong_caption_pl: wrong_caps,
                    wrong_image_pl: wrong_images
                }
            )

            tb_writer.add_summary(summary=summary, global_step=epoch)

    ##########
    # end training loop
    # save final weights to load into for GAN
    rnn_weights_file = os.path.join(os.getcwd(), "rnn_weights.h5")
    rnn.save_weights(rnn_weights_file)


############################################################
# GAN
def train_gan():
    print("\nTraining GAN...")

    # create folder to save checkpoints
    save_folder = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(1)  # 1=train, 0=test

    ##########
    # Instantiate models
    rnn = build_rnn()
    rnn.load_weights("rnn_weights.h5")

    ##########
    # Placeholders
    real_image_pl = tf.placeholder(dtype=tf.float32,
                                   shape=[None, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS],
                                   name="real_image_pl")

    real_caption_pl = tf.placeholder(dtype=tf.float32,
                                     shape=[None, None],
                                     name="real_caption_pl")

    wrong_caption_pl = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None],
                                      name="wrong_caption_pl")

    noise_pl = tf.placeholder(dtype=tf.float32,
                              shape=[None, Z_DIM],
                              name="noise_pl")

    g_out = build_generator(noise_pl, rnn(real_caption_pl), reuse=False)
    d_real_out = build_discriminator(real_image_pl, rnn(real_caption_pl), reuse=False)
    d_fake_out = build_discriminator(g_out, rnn(real_caption_pl), reuse=True),
    d_mismatch_out = build_discriminator(real_image_pl, rnn(wrong_caption_pl), reuse=True)

    ##########
    # Loss function
    d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real_out,
        labels=tf.ones_like(d_real_out)
    ))

    d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_out,
        labels=tf.zeros_like(d_fake_out)
    ))

    d_mismatch_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_mismatch_out,
        labels=tf.zeros_like(d_mismatch_out)
    ))

    d_loss = d_real_loss + 0.5 * (d_fake_loss + d_mismatch_loss)

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_out,
        labels=tf.ones_like(d_fake_out)
    ))

    ##########
    # Optimizer
    d_optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE,
        beta1=0.5
    ).minimize(
        loss=d_loss,
        var_list=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")]
    )

    g_optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE,
        beta1=0.5
    ).minimize(
        loss=g_loss,
        var_list=[tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")]
    )

    ##########
    # Session initialization and TensorBoard Setup
    sess = tf.Session()
    tf.summary.scalar(name="D Loss", tensor=d_loss)
    tf.summary.scalar(name="G Loss", tensor=g_loss)
    tf.summary.image(name="Generated Images",
                     tensor=build_generator(noise_pl, rnn(real_caption_pl), reuse=True),
                     max_outputs=20)
    tb = tf.summary.merge_all()
    tb_writer = tf.summary.FileWriter(logdir=save_folder, graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    ##########
    # training loop
    for epoch in range(NUM_EPOCHS+1):
        # TRAIN DISCRIMINATOR
        # right captions
        cap_idx = get_random_int(min=0, max=n_captions_train - 1, number=BATCH_SIZE)
        real_caps = captions_ids_train[cap_idx]
        real_caps = pad_sequences(real_caps, maxlen=128, padding="post")  # pad captions to fixed length

        # right images
        img_idx = np.floor(np.asarray(cap_idx).astype("float") / n_captions_per_image).astype("int")
        real_images = images_train[img_idx]

        # wrong caption
        cap_idx = get_random_int(min=0, max=n_captions_train - 1, number=BATCH_SIZE)
        wrong_caps = captions_ids_train[cap_idx]
        wrong_caps = pad_sequences(wrong_caps, maxlen=128, padding="post")

        # noise
        noise = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise

        # preprocessing on images
        real_images = threading_data(real_images, prepro_img, mode="train")

        #
        d_error, _ = sess.run(
            [d_loss, d_optimizer],
            feed_dict={
                real_image_pl: real_images,
                real_caption_pl: real_caps,
                wrong_caption_pl: wrong_caps,
                noise_pl: noise
            }
        )

        # TRAIN GENERATOR
        # right captions
        cap_idx = get_random_int(min=0, max=n_captions_train - 1, number=BATCH_SIZE)
        real_caps = captions_ids_train[cap_idx]
        real_caps = pad_sequences(real_caps, maxlen=128, padding="post")  # pad captions to fixed length

        # noise
        noise = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise

        g_error, _ = sess.run(
            [g_loss, g_optimizer],
            feed_dict={
                real_caption_pl: real_caps,
                noise_pl: noise
            }
        )

        #
        if epoch % 20 == 0:
            print("\nEpoch: {}".format(epoch))
            print("D ERROR: {}, G ERROR: {}".format(d_error, g_error))

        # generate image to TensorBoard
        if epoch % 100 == 0:
            # right captions
            cap_idx = get_random_int(min=0, max=n_captions_train - 1, number=BATCH_SIZE)
            real_caps = captions_ids_train[cap_idx]
            real_caps = pad_sequences(real_caps, maxlen=128, padding="post")  # pad captions to fixed length

            # noise
            noise = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise

            summary = sess.run(
                tb,
                feed_dict={
                    real_caption_pl: real_caps,
                    real_image_pl: real_images,
                    wrong_caption_pl: wrong_caps,
                    noise_pl: noise
                }
            )

            tb_writer.add_summary(summary=summary, global_step=epoch)

        # save image
        if epoch % 1000 == 0:
            # right captions
            cap_idx = get_random_int(min=0, max=n_captions_train - 1, number=BATCH_SIZE)
            real_caps = captions_ids_train[cap_idx]
            real_caps = pad_sequences(real_caps, maxlen=128, padding="post")  # pad captions to fixed length

            # noise
            noise = np.random.normal(size=(BATCH_SIZE, Z_DIM))  # Gaussian noise

            img_idx = 0
            gen_img, _ = sess.run(
                [
                    build_generator(noise_pl, rnn(real_caption_pl), reuse=True),
                    rnn(real_caption_pl)
                ],
                feed_dict={
                    real_caption_pl: real_caps,
                    noise_pl: noise
                }
            )

            #gen_img = (gen_img + 1.0) / 2.0  # [-1,1] => [0,1]
            gen_img = threading_data(gen_img, prepro_img, mode="rescale")

            row = 8
            col = 8

            figure = np.zeros((64*row, 64*col, 3))
            for i in range(row):
                for j in range(col):
                    figure[i*64: (i+1)*64, j*64: (j+1)*64, :] = gen_img[img_idx]
                    img_idx += 1

            plt.imshow(figure)
            plt.savefig(save_folder + "\\generated_" + str(epoch) + ".png")


############################################################
# MAIN
if __name__ == "__main__":
    if "--rnn" in sys.argv:
        train_rnn()

    if "--gan" in sys.argv:
        train_gan()
