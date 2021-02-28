import numpy as np
import matplotlib.pyplot as plt
import idx2numpy 

### Feb 24 2021 (last modified on Feb 25)
### Arthur Deng
### This module is for loading/preprocessing data
### Future changes: add feature scaling/mean normalization

def convert_to_numpy(filename):
    """ reads from file
    """
    f_read = open(filename, 'rb')
    ndarr = idx2numpy.convert_from_file(f_read)
    return ndarr

def feature_scaling(data, m, n):
    scaled = [np.zeros(n) for i in range(m)]
    for feature in range(n):
        # compute mean, min, and max
        sum, mean, MIN, MAX = 0, 0, data[0][feature], data[0][feature]
        for i in range(m):
            sum += data[i][feature]
            MIN = min(MIN, data[i][feature])
            MAX = max(MAX, data[i][feature])
        mean, s = sum/m, MAX - MIN
        if s == 0:
            continue
        for i in range(m):
            scaled[i][feature] = (data[i][feature] - mean) / s
    return scaled


def initialize():
    """ function to load data
        m = 60000
        train_img_raw is a 60000 x 28 x 28 array; contains x(i) (28 x 28 matrix)
        train_label_raw is a 60000 x 1 array; contains y(i) | {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    """

    # read from file
    train_img_raw = convert_to_numpy('Data/train-img')
    test_img_raw = convert_to_numpy('Data/test-img')
    train_result = convert_to_numpy('Data/train-label') 
    test_result = convert_to_numpy('Data/test-label')

    train_len, test_len, dim = len(train_img_raw), len(test_img_raw), 28

    # x(i) is a 28^2 + 1 dimension vector, index 0 is x0, the intercept term
    train_img = [train_img_raw[i].reshape([dim**2]) for i in range(train_len)]
    test_img = [test_img_raw[i].reshape([dim**2]) for i in range(test_len)]
    
    # feature scaling
    # train_img = feature_scaling(train_img, train_len, dim**2)
    # test_img = feature_scaling(test_img, test_len, dim**2)
    
    # y(i) is a 10 dimension vector
    train_label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(train_len)]
    test_label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(test_len)]

    for i in range(train_len):
        train_img[i] = np.append([1], train_img[i]) # add the intercept term
        train_label[i][train_result[i]] = 1
    for i in range(test_len):
        test_img[i] = np.append([1], test_img[i]) # add the intercept term
        test_label[i][test_result[i]] = 1

    # splitting test set to 50% test and 50% cross validation
    CV_label, CV_img = test_label[test_len//2:], test_img[test_len//2:]
    test_label, test_img = test_label[:test_len//2], test_img[:test_len//2]

    #returns three sets: train, CV, and test
    return train_img, train_label, CV_img, CV_label, test_img, test_label

