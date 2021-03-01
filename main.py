import numpy as np
import matplotlib.pyplot as plt
import load_data, gradient_descent, matrix_file
### Last Modified: Mar 1, 2021

#train_img, train_label, CV_img, CV_label, test_img, test_label = load_data.initialize()
train_img, train_label = load_data.initialize()
### Implement neural network, forward propagation, cost function here. 
### This is the file to run.

### L = 3
n = 785
m = 60000
m_dev = 5 #smaller number of training data to check functionality of program
regularization_parameter = -1 #lambda -- regularization parameter (Bias/Variance tradeoff)
cost = 0
L = 3

def sigmoid(z):
    """ Sigmoid Function
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    """ Derivative of Sigmoid Function
        ATTENTION: a is a vector instead of a scalar as in the sigmoid() function
    """
    return np.dot(a, (np.ones(a.shape) - a))

def J(h, y):
    """ Cost function for hypothesis h(theta)
    """
    c = 0
    for k in range(len(h)):
        c -= y[k]*np.log(h[k])+(1-y[k])*np.log(1-h[k])
    return c

### Tranfer Matrices with randomly initialized weights
### Theta[j] maps layer j to layer j + 1
Theta = [np.zeros((n-1, n)), np.zeros((10, n))]
### initial randomized values of the array stored in theta.txt
### read() loads the values

# matrix_file.write(Theta)
Theta = matrix_file.read(Theta)

def forward_prop(x, y, Theta):
    """ Forward Propagation
        x: train_img[i]
        Theta: Theta
        y: train_label[i]
        return: a[L-1] --- h(x)
    """
    a = [[] for j in range(L)]
    a[0] = x
    for j in range(L - 1):
        a[j+1] = np.dot(Theta[j], a[j])
        ### not forgetting to append the constant intercept term
        if j+1 < L-1:
            a[j+1] = np.append([1], a[j+1])
        for k in range(1, len(a[j+1])):
            a[j+1][k] = sigmoid(a[j+1][k])
    return a

def test():
    """ for testing functionality of forward propagation
    """
    for i in range(m_dev):
        h = forward_prop(train_img[i], train_label[i], Theta)
        cost += J(h, train_label[i])
    print(round(cost, 2))
    
def backprop(X, Y, Theta):
    """ Backprop Algorithm
        X: train_img
        Y: train_label
        Theta: initial value of Theta
        return: partial derivatives for Theta (same shape as Theta)
    """

    DELTA = [np.zeros(Theta[i].shape[0], Theta[i].shape[1]) for i in range(L-1)]
    D = [np.zeros(Theta[i].shape[0], Theta[i].shape[1]) for i in range(L-1)]
    for i in range(m_dev):
        ### construct delta, index 0 should always be []
        delta, a = [[] for i in range(L)], forward_prop(X[i], Y[i], Theta)
        
        delta[L - 1] = a[L - 1] - Y[i]
        for k in range(L-2, 0, -1):
            delta[k] = np.dot(np.dot(np.transpose(Theta[k]), delta[k+1]), 
                                sigmoid_derivative(a[k]))





