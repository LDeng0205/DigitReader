
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import load_data

train_img, train_label = load_data.initialize()

plt.matshow(train_img[1][1:].reshape((28, 28)))
plt.show()
print(train_label[1])

# # implementation of gradient descent

# cost = []
# def gradient_descent(
#     gradient, start, learn_rate, n_iter=50, tolerance=1e-06
# ):
#     vector = start
#     for _ in range(n_iter):
#         diff = -learn_rate * gradient(vector)
#         #check if step is less than minimum
#         if np.all(np.abs(diff) <= tolerance):
#             break
#         vector += diff
#         #plotting
#         cost.append(vector)
#     return vector

# print(gradient_descent(gradient = lambda x:2*x, start = 10.0, learn_rate = 0.2))

# plt.plot(cost)
# plt.ylabel('some numbers')
# plt.show()