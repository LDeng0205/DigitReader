import main
import numpy as np
# CV_img, CV_label, test_img, test_label
# print(main.Theta)
l = len(main.CV_img)
correct = 0
print("CV length: ", l)
for i in range(l):
    if main.predict(main.Theta, main.CV_img[i]) == np.argmax(main.CV_label[i]):
        correct += 1
precision = correct / l
print("CV: ", precision)

true_positives = 0
for i in range(len(main.test_img)): 
    if main.predict(main.Theta, main.test_img[i]) == np.argmax(main.test_label[i]):
        true_positives += 1
print("result: ", true_positives/len(main.test_img))
