"""
Michael Patel
August 2018

Python 3.6.5
TensorFlow 1.12.0

Project description:
    To synthesize images from text-based captions as a form of reverse image captioning.

File description:
    To run preprocessing and training algorithms

Dataset: Oxford-102 flowers, http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

"""
################################################################################
# Imports
import os
import sys
import argparse
import numpy as np
import pickle
from datetime import date
import matplotlib as plt

import tensorflow as tf

from model import build_cnn, build_rnn


################################################################################
# Train RNN
def train_rnn():
    print("\nTraining RNN...")


################################################################################
# Train GAN
def train_gan():
    print("\nTraining GAN...")

    pickle_filepath = os.path.join(os.getcwd(), "data/pickle data")

    with open(os.path.join(pickle_filepath, "_vocab.pickle"), 'rb') as f:
        vocab = pickle.load(f)

    with open(os.path.join(pickle_filepath, "_image_train.pickle"), 'rb') as f:
        _, images_train = pickle.load(f)

    with open(os.path.join(pickle_filepath, "_image_test.pickle"), 'rb') as f:
        _, images_test = pickle.load(f)

    with open(os.path.join(pickle_filepath, "_n.pickle"), 'rb') as f:
        n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)

    with open(os.path.join(pickle_filepath, "_caption.pickle"), 'rb') as f:
        captions_ids_train, captions_ids_test = pickle.load(f)


################################################################################
# MAIN
if __name__ == "__main__":
    # enable eager execution
    tf.enable_eager_execution()

    # print out TF version
    print("TF version: {}".format(tf.__version__))

    # CLI Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--rnn", help="Train RNN", action="store_true")
    parser.add_argument("--gan", help="Train GAN", action="store_true")

    args = parser.parse_args()

    if args.rnn:
        train_rnn()

    elif args.gan:
        train_gan()

    else:
        print("\nPlease provide an argument: ")
        parser.print_help()
        sys.exit(1)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
