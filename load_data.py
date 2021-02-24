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
    trainImg = convert_to_numpy('Data/train-img') 
    trainLabel = convert_to_numpy('Data/train-label') 
    testImg = convert_to_numpy('Data/test-img')
    testLabel = convert_to_numpy('Data/test-label')
    return trainImg, trainLabel, testImg, testLabel


