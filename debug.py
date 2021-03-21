import numpy as np
import matplotlib.pyplot as plt
import load_data, matrix_file

### Last Modified: Mar 1, 2021

#train_img, train_label, CV_img, CV_label, test_img, test_label = load_data.initialize()
train_img, train_label = load_data.initialize()
### Implement neural network, forward propagation, cost function here. 
### This is the file to run.

n = 785
m = 60000
m_dev = 1 #smaller number of training data to check functionality of program
lam = 0 #lambda -- regularization parameter (Bias/Variance tradeoff)
L = 4 # layers: layer 0, layer 1, layer 2

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
    """ Cost function for hypothesis h(theta)
    """
    c = 0
    for k in range(len(h)):
        c += -(y[k]*np.log(h[k])+(1-y[k])*np.log(1-h[k]))
    return c

def gradient_checking(Theta, X = train_img, Y = train_label):
    """ gradient checking
        uses X[0] and Y[0] by default for approximation
    """
    epsilon= 0.0001
    D_check = [np.zeros((Theta[i].shape[0], Theta[i].shape[1])) for i in range(L)]
    for l in range(L):
        row, col = D_check[l].shape[0], D_check[l].shape[1]
        # print(row, col)
        for i in range(row):
            for j in range(col):    
                Theta[l][i][j] += epsilon
                thetaPlus = forward_prop(Theta, X[0], Y[0])
                Theta[l][i][j] -= 2 * epsilon
                thetaMinus = forward_prop(Theta, X[0], Y[0])
                Theta[l][i][j] += epsilon
                # if l == 1:
                #     print((J(thetaPlus, X[0], Y[0]) - J(thetaMinus, X[0], Y[0]))/(2*epsilon))
                D_check[l][i][j] = (J(thetaPlus[2], Y[0]) - J(thetaMinus[2], Y[0])) / (2 * epsilon)
    return (D_check)
    # print(D[1])
    # plt.matshow(D[0].reshape((60, 131)))
    # plt.matshow(D_check[0].reshape((60, 131)))
    # plt.matshow(D_check[1])
    # plt.show()
    # n = (D[1]-D_check[1]).reshape((10, 11))
    # plt.matshow(n)
    # plt.show()
    # cnt = 0
    # for i in range(D[0].shape[0]):
    #     for j in range(D[0].shape[1]):
    #         if n[i][j] > D[0][i][j]:
    #             cnt += 1
    #             

    # print(cnt)


def forward_prop(Theta, x, y):
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

def backprop(Theta, X = train_img, Y = train_label, m_dev = m_dev):
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
        delta = [[] for _ in range(L+1)]
        a, z, h =  forward_prop(Theta, X[i], Y[i])
        delta[L] = np.hstack(([1], a[L] - Y[i]))
        for k in reversed(range(1, L)):
            if k == L - 1:
                delta[k] = np.multiply(np.dot(Theta[k].T, delta[k+1][1:]), 
                                sigmoid_derivative(a[k]))
            else:
                delta[k] = np.multiply(np.dot(Theta[k].T, delta[k+1][1:]), 
                                np.hstack(([1], sigmoid_derivative(a[k][1:]))))
            print(delta)
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
                    D[l][i][j] = 1/m_dev * DELTA[l][i][j]
                else: 
                    # add regularization
                    D[l][i][j] = 1/m_dev * DELTA[l][i][j] + lam * Theta[l][i][j]
    return D




test_theta = [np.random.rand(2, 3), np.random.rand(2, 3), np.random.rand(2, 3), np.random.rand(2, 3)]
test_x_set = [np.array([2, 3])]
test_y_set = [np.array([4, 5])]
m_dev = 1
D = backprop(test_theta, X = test_x_set, Y = test_y_set, m_dev = 1)
print(D)
print("====")
print(gradient_checking(test_theta, X = test_x_set, Y = test_y_set))