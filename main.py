import numpy as np
import matplotlib.pyplot as plt
import load_data, matrix_file
### Last Modified: Mar 1, 2021

train_img, train_label, CV_img, CV_label, test_img, test_label = load_data.initialize()
# train_img, train_label = load_data.initialize()
### Implement neural network, forward propagation, cost function here. 
### This is the file to run.

n = 785
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
def J_total(Theta, X, Y, m):
    """ Cost function for the training set size m
    """
    cost, reg = 0, 0
    for i in range(m):
        a, z, h = forward_prop(Theta, X[i])
        cost += J(h, Y[i])
    cost /= m
    for l in range(len(Theta)):
        for i in range(Theta[l].shape[0]):
            for j in range(Theta[l].shape[1]):
                reg += Theta[l][i][j] ** 2
    reg *= lam / (2 * m)
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

def train(Theta, m, t, X = train_img, Y = train_label):
    """ training neural network
    """
    max_iter, iter, tolerance = t, 0, 1e-06
    graph = [] # for graphing cost
    while iter < max_iter:
        print("loop count: ", iter)
        DELTA = [np.zeros((Theta[i].shape[0], Theta[i].shape[1])) for i in range(L)]
        for i in range(m):
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
                        DELTA[l][i][j] = 1/m * DELTA[l][i][j]
                    else: 
                        # add regularization
                        DELTA[l][i][j] = 1/m * DELTA[l][i][j] + lam * Theta[l][i][j]
        # DELTA are the matrices of partial derivatives
        cont = update(DELTA, Theta, tolerance)
        
        graph.append(J_total(Theta, X, Y, m))
        iter += 1
    plt.plot(graph)
    plt.show()
    return Theta
def update(PD, Theta, tolerance, LR = learning_rate):
    cont = False
    for l in range(len(Theta)):
        for i in range(Theta[l].shape[0]):
            for j in range(Theta[l].shape[1]):
                Theta[l][i][j] = Theta[l][i][j] - LR * PD[l][i][j]
                if (abs(LR * PD[l][i][j]) > tolerance):
                    cont = True
    return cont
def predict(Theta, x):
    a, z, h = forward_prop(Theta, x)
    # print(np.argmax(h))
    return np.argmax(h)


matrix_file.write(train(Theta, m = 10000, t = 2), trained = True)



