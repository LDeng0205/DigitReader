import numpy as np
import matplotlib.pyplot as plt
import load_data, backprop, gradient_descent, matrix_file
#Last Modified: Feb 28, 2021

train_img, train_label, CV_img, CV_label, test_img, test_label = load_data.initialize()

### Implement neural network, forward propagation, cost function here. 
### This is the file to run.

### L = 3
n = 785
m = 60000
m_dev = 10 #smaller number of training data to check functionality of program

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Tranfer Matrices with randomly initialized weights
# Theta[j] maps layer j to layer j + 1
Theta = [np.zeros((n, n)), np.zeros((10, n))]
# initial randomized values of the array stored in theta.txt
# read() loads the values

# matrix_file.write(Theta)
Theta = matrix_file.read(Theta)


# Forward Propagation
for i in range(m_dev):
    a = [[] for j in range(len(Theta) + 1)]
    a[0] = train_img[i]
    for j in range(len(Theta)):
        a[j + 1] = np.dot(Theta[j], a[j])
        for k in range(len(a[j+1])):
            a[j+1][k] = sigmoid(a[j+1][k])
    print(a[2])
    



