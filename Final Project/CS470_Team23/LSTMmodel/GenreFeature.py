import librosa
import math
import os
import re

import numpy as np


class GenreFeature:

    hop_length = None
    genre_list = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

    relative_path = "./dataset/"
    dir_trainfolder = relative_path + "_train"
    dir_devfolder = relative_path + "_validation"
    dir_testfolder = relative_path + "_test"
    dir_all_files = relative_path + ""

    train_X_preprocessed_data = relative_path + "data_train_input.npy"
    train_Y_preprocessed_data = relative_path + "data_train_target.npy"
    valid_X_preprocessed_data = relative_path + "data_valid_input.npy"
    valid_Y_preprocessed_data = relative_path + "data_valid_target.npy"
    test_X_preprocessed_data = relative_path + "data_test_input.npy"
    test_Y_preprocessed_data = relative_path + "data_test_target.npy"

    train_X = train_Y = None
    valid_X = valid_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.hop_length = 512
        
    def load_deserialize_data(self):

        self.train_X = np.load(self.train_X_preprocessed_data)
        self.train_Y = np.load(self.train_Y_preprocessed_data)

        self.valid_X = np.load(self.valid_X_preprocessed_data)
        self.valid_Y = np.load(self.valid_Y_preprocessed_data)

        self.test_X = np.load(self.test_X_preprocessed_data)
        self.test_Y = np.load(self.test_Y_preprocessed_data)
