import numpy as np
import matplotlib.pyplot as plt
import load_data, gradient_descent, matrix_file
### Last Modified: Mar 1, 2021

#train_img, train_label, CV_img, CV_label, test_img, test_label = load_data.initialize()
train_img, train_label = load_data.initialize()
### Implement neural network, forward propagation, cost function here. 
### This is the file to run.

n = 785
m = 60000
m_dev = 50 #smaller number of training data to check functionality of program
lam = 0 #lambda -- regularization parameter (Bias/Variance tradeoff)
L = 2 # layers: layer 0, layer 1, layer 2

def sigmoid(z):
    """ Sigmoid Function
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    """ Derivative of Sigmoid Function
        ATTENTION: a is a vector instead of a scalar as in the sigmoid() function
    """
    return np.dot(a, (np.ones(a.shape) - a))

def J(Theta, x, y):
    """ Cost function for hypothesis h(theta)
    """
    c = 0
    h = forward_prop(Theta, x, y)[L]
    for k in range(len(h)):
        c += -(y[k]*np.log(h[k])+(1-y[k])*np.log(1-h[k]))
    return c

def gradient_checking(Theta, X = train_img, Y = train_label):
    """ gradient checking
        uses X[0] and Y[0] by default for approximation
    """
    epsilon, D = 0.1, backprop(Theta)
    D_check = [np.zeros((Theta[i].shape[0], Theta[i].shape[1])) for i in range(L)]
    for l in range(L):
        row, col = D_check[l].shape[0], D_check[l].shape[1]
        # print(row, col)
        for i in range(row):
            for j in range(col):    
                thetaMinus = [Theta[i].copy() for i in range(len(Theta))]
                thetaPlus = [Theta[i].copy() for i in range(len(Theta))]
                thetaMinus[l][i][j] -= epsilon
                thetaPlus[l][i][j] += epsilon
                # if l == 1:
                #     print((J(thetaPlus, X[0], Y[0]) - J(thetaMinus, X[0], Y[0]))/(2*epsilon))
                D_check[l][i][j] = (J(thetaPlus, X[0], Y[0]) - J(thetaMinus, X[0], Y[0])) / (2 * epsilon)
    
    print(D_check[1])
    # print(D[1])
    plt.matshow(D[0].reshape((50, 157)))
    plt.matshow(D_check[0].reshape((50, 157)))
    n = (D[0]-D_check[0]).reshape((50, 157))
    # plt.matshow(n)
    plt.show()
    # cnt = 0
    # for i in range(D[0].shape[0]):
    #     for j in range(D[0].shape[1]):
    #         if n[i][j] > D[0][i][j]:
    #             cnt += 1
    #             # print("fuck")

    # print(cnt)
### Tranfer Matrices with randomly initialized weights
### Theta[j] maps layer j to layer j + 1
Theta = [np.zeros((10, n)), np.zeros((10, 11))]
### initial randomized values of the array stored in theta.txt
### read() loads the values

# matrix_file.write(Theta)
Theta = matrix_file.read(Theta)

def forward_prop(Theta, x, y):
    """ Forward Propagation
        Theta: Theta
        X: train_img[i]
        Y: train_label[i]
        return: a -- vector
    """
    a = [[] for j in range(L+1)]
    a[0] = x
    for j in range(L):
        a[j+1] = np.dot(Theta[j], a[j])
        ### not forgetting to append the constant intercept term
        
        for k in range(len(a[j+1])):
            a[j+1][k] = sigmoid(a[j+1][k])
            # if (a[j+1][k]<0):
            #     print("FUCK")
        if j+1 < L:
            a[j+1] = np.append([1], a[j+1])
    return a
    
def backprop(Theta, X = train_img, Y = train_label):
    """ Backprop Algorithm
        X: train_img
        Y: train_label
        Theta: initial value of Theta
        return: partial derivatives for Theta (same shape as Theta)
    """

    DELTA = [np.zeros((Theta[i].shape[0], Theta[i].shape[1])) for i in range(L)]
    D = [np.zeros((Theta[i].shape[0], Theta[i].shape[1])) for i in range(L)]
    for i in range(m_dev):
        ### construct delta, index 0 should always be []
        delta, a = [[] for i in range(L+1)], forward_prop(Theta, X[i], Y[i])
        delta[L] = a[L] - Y[i]
        for k in range(L-1, 0, -1):
            delta[k] = np.multiply(np.dot(np.transpose(Theta[k]), delta[k+1]), 
                                sigmoid_derivative(a[k]))
        for l in range(L-1):
            for rowidx in range(DELTA[l].shape[0]):
                for colidx in range(DELTA[l].shape[1]):
                    DELTA[l][rowidx][colidx] += a[l][colidx] * delta[l+1][rowidx]
    for l in range(L):
        row, col = DELTA[l].shape[0], DELTA[l].shape[1]
        for i in range(row):
            for j in range(col):
                if j == 0:
                    # no regularization term for the first row
                    D[l][i][j] = 1/m_dev * DELTA[l][i][j]
                else: 
                    # add regularization
                    D[l][i][j] = 1/m_dev * DELTA[l][i][j] + lam * Theta[l][i][j]
    return D
        
gradient_checking(Theta)

# for i in range(m_dev):
#     a = forward_prop(Theta, train_img[i], train_label[i])
#     # print(a[L-1])
#     print(J(Theta, train_img[i], train_label[i]))
# print(backprop(Theta))
# backprop(Theta)


