#
# Notes:
#

############################################################
# IMPORTS
import tensorflow as tf

import os
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


############################################################
# HYPERPARAMETERS and DESIGN CHOICES


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

images_train = np.array(images_train)
images_test = np.array(images_test)
