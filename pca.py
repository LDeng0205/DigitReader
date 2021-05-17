import numpy as np
import matplotlib.pyplot as plt
import idx2numpy 

### May 17 2021 (last modified on Feb 28)
### Arthur Deng
### This module is for dimensionality reduction

def convert_to_numpy(filename):
    """ reads from file
    """
    f_read = open(filename, 'rb')
    ndarr = idx2numpy.convert_from_file(f_read)
    return ndarr

### read from file
train_img_raw = convert_to_numpy('Data/train-img')
test_img_raw = convert_to_numpy('Data/test-img')
train_result = convert_to_numpy('Data/train-label') 
test_result = convert_to_numpy('Data/test-label')

train_len, test_len, dim = len(train_img_raw), len(test_img_raw), 28

# train_len //= 6 # for faster processing during development
# test_len //= 10
### x(i) is a 28^2 + 1 dimension vector, index 0 is x0, the intercept term
train_img = [train_img_raw[i].reshape([dim**2]) for i in range(train_len)]
test_img = [test_img_raw[i].reshape([dim**2]) for i in range(test_len)]

A = []
idx = [0]
for num in range(10):
    cnt = 0
    for i in range(len(train_img)):
        if train_result[i] == num:
            A.append(train_img[i])
            cnt += 1
    idx.append(idx[num] + cnt)

print(idx)
# print(A[0].shape)


u, s, vt = np.linalg.svd(A, full_matrices=False)
print(u.shape)
print(s.shape)
print(vt.shape)
# 10 singular values
sv = 10
new_basis = vt[0:sv, :].T
print(new_basis)

projected_A = np.dot(A, new_basis)

print(projected_A.shape)
plt.plot(s[:10])
plt.show()

#==================================================================================

import numpy as np
import matplotlib.pyplot as plt
import load_data, matrix_file
from datetime import datetime
### Last Modified: Mar 1, 2021

print("Launched: ", datetime.now().strftime("%H:%M:%S"))
train_img, train_label, CV_img, CV_label, test_img, test_label = load_data.initialize()
print("Finished loading and processing data: ", datetime.now().strftime("%H:%M:%S"))
# train_img, train_label = load_data.initialize()
### Implement neural network, forward propagation, cost function here. 
### This is the file to run.

n = 10
m = len(train_img)
m_CV = len(CV_img)
m_test = len(test_img)
lam = 0.01 #lambda -- regularization parameter (Bias/Variance tradeoff)
L = 2 # layers: layer 0, layer 1, layer 2
learning_rate = 0.1
### Tranfer Matrices with randomly initialized weights
### Theta[j] maps layer j to layer j + 1
Theta = [np.zeros((10, n + 1)), np.zeros((10, 11))]
### initial randomized values of the array stored in theta.txt
### read() loads the values

# matrix_file.write(Theta)
Theta = matrix_file.read(Theta)


def sigmoid(z):
    """ Sigmoid Function
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    """ Derivative of Sigmoid Function
        ATTENTION: takes in 'a' instead of 'z'
    """
    return a * (1 - a)

def J(h, y):
    """ Cost function for one hypothesis h(theta)
    """
    c = 0
    for k in range(len(h)):
        c += -(y[k]*np.log(h[k])+(1-y[k])*np.log(1-h[k]))
    return c
def J_total(Theta, X, Y, m_start, m_end):
    """ Cost function for the training set size m
    """
    cost, reg = 0, 0
    for i in range(m_start, m_end):
        a, z, h = forward_prop(Theta, X[i])
        cost += J(h, Y[i])
    cost /= (m_end - m_start)
    for l in range(len(Theta)):
        for i in range(Theta[l].shape[0]):
            for j in range(Theta[l].shape[1]):
                reg += Theta[l][i][j] ** 2
    reg *= lam / (2 * (m_end - m_start))
    return cost + reg

def forward_prop(Theta, x):
    """ Forward Propagation
        Theta: Theta
        X: train_img[i]
        Y: train_label[i]
        return: A -- results from each layer
    """
    A = [np.hstack(([1], x))]       
    Z = []
    for idx, theta in enumerate(Theta):
        Z.append(np.matmul(theta, A[-1]))
        # adding bias column to the output of previous layer
        A.append(np.hstack(([1], sigmoid(Z[-1]))))
    # bias is not needed in the final output
    A[-1] = A[-1][1:]
    y_hat = A[-1]
    return A, Z, y_hat

def back_prop(Theta, x, y):
    """ Backprop Algorithm
        x: train_img[i]
        y: train_label[i]
        Theta: weights
        return: delta; error associated with each neuron
    """
    delta = [[] for _ in range(L+1)]
    a, z, h =  forward_prop(Theta, x)
    delta[L] = np.hstack(([1], a[L] - y))
    for k in reversed(range(1, L)):
        if k == L - 1:
            delta[k] = np.multiply(np.dot(Theta[k].T, delta[k+1][1:]), 
                            sigmoid_derivative(a[k]))
        else:
            delta[k] = np.multiply(np.dot(Theta[k].T, delta[k+1][1:]), 
                            np.hstack(([1], sigmoid_derivative(a[k][1:]))))
    return a, z, h, delta

def train(Theta, m_start, m_end, t, X = train_img, Y = train_label):
    """ training neural network
    """
    max_iter, iter, tolerance = t, 0, 1e-06
    graph = [] # for graphing cost
    print(datetime.now().strftime("%H:%M:%S"))
    while iter < max_iter + 1:
        print("loop count: ", iter, "/", max_iter)
        DELTA = [np.zeros((Theta[i].shape[0], Theta[i].shape[1])) for i in range(L)]
        for i in range(m_start, m_end):
            ### construct delta, index 0 should always be []
            a, z, h, delta = back_prop(Theta, X[i], Y[i])
            for l in range(L):
                for rowidx in range(DELTA[l].shape[0]):
                    for colidx in range(DELTA[l].shape[1]):
                        # don't use delta[l+1][0] because that term is always for the regularization parameter
                        DELTA[l][rowidx][colidx] += a[l][colidx] * delta[l+1][rowidx + 1]
            
        for l in range(L):
            row, col = DELTA[l].shape[0], DELTA[l].shape[1]
            for i in range(row):
                for j in range(col):
                    if j == 0:
                        # no regularization term for the first row
                        DELTA[l][i][j] = 1/(m_end - m_start) * DELTA[l][i][j]
                    else: 
                        # add regularization
                        DELTA[l][i][j] = 1/(m_end - m_start) * DELTA[l][i][j] + lam * Theta[l][i][j]
        # DELTA are the matrices of partial derivatives
        cont = update(DELTA, Theta, tolerance)
        if (not cont):
            break
        if iter % 10 == 0:
            matrix_file.write(Theta, trained = True)
            print("Weights saved: ", iter)
            
            J = J_total(Theta, X, Y, m_start, m_end)
            print("cost: ", J)
            graph.append(J)
        iter += 1
    print("Batch finished: ", datetime.now().strftime("%H:%M:%S"), f' (for samples {m_start} to {m_end})')
    return graph
def update(PD, Theta, tolerance, LR = learning_rate):
    """ Updating Theta after each iteration of gradient descent
    """
    cont = False
    for l in range(len(Theta)):
        for i in range(Theta[l].shape[0]):
            for j in range(Theta[l].shape[1]):
                Theta[l][i][j] = Theta[l][i][j] - LR * PD[l][i][j]
                if (abs(LR * PD[l][i][j]) > tolerance):
                    cont = True
    return cont
def predict(Theta, x):
    """ Predict result based on hypothesis
    """
    a, z, h = forward_prop(Theta, x)
    # print(np.argmax(h))
    return np.argmax(h)


