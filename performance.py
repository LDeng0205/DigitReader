import main
import numpy as np
# CV_img, CV_label, test_img, test_label
# print(main.Theta)
l = len(main.CV_img)
correct = 0
for i in range(l):
    if main.predict(main.Theta, main.CV_img[i]) == np.argmax(main.CV_label[i]):
        correct += 1
precision = correct / l
print(precision)