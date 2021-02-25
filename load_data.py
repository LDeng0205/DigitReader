import numpy as np
import matplotlib.pyplot as plt
import idx2numpy 

### Feb 24 2021 
### Arthur Deng
### This module is for loading/preprocessing data
### Future changes: add feature scaling/mean normalization

def convert_to_numpy(filename):
    f_read = open(filename, 'rb')
    ndarr = idx2numpy.convert_from_file(f_read)
    return ndarr


def initialize():
    # m = 60000
    # trainImg is a 60000 x 28 x 28 array; contains x(i) (28 x 28 matrix)
    # trainLabel is a 60000 x 1 array; contains y(i) | {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    # read from file
    train_img_raw = convert_to_numpy('Data/train-img')
    test_img_raw = convert_to_numpy('Data/test-img')
    train_result = convert_to_numpy('Data/train-label') 
    test_result = convert_to_numpy('Data/test-label')

    train_len, test_len, dim = len(train_img_raw), len(test_img_raw), 28

    # x(i) is a 28^2 dimension vector
    train_img = [train_img_raw[i].reshape([1, dim**2]) for i in range(train_len)]
    test_img = [test_img_raw[i].reshape([1, dim**2]) for i in range(test_len)]

    # y(i) is a 10 dimension vector
    train_label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(train_len)]
    test_label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(test_len)]
    for i in range(train_len):
        train_label[i][train_result[i]] = 1
    for i in range(test_len):
        test_label[i][test_result[i]] = 1

    # splitting test set to 50% test and 50% cross validation
    CV_label, CV_img = test_label[test_len//2:], test_img[test_len//2:]
    test_label, test_img = test_label[:test_len//2], test_img[:test_len//2]

    #returns three sets: train, CV, and test
    return train_img, train_label, CV_img, CV_label, test_img, test_label


initialize()