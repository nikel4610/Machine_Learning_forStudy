import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

def testimg():
    img = cv2.imread('D:/vsc_project/machinelearning_study/test.jpg', cv2.IMREAD_GRAYSCALE)
    # edge1 = cv2.Canny(img, 50, 150)
    edge2 = cv2.Canny(img, 100, 200)
    # cv2.imshow('edge1', edge1)
    cv2.imshow('img', img)
    cv2.imshow('edge2', edge2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testimg2():
    img = cv2.imread('D:/vsc_project/machinelearning_study/test.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()

# # 이미지를 컨볼루션 후 plt로 출력
# def testimg3():
#     img = cv2.imread('D:/vsc_project/machinelearning_study/test.jpg', cv2.IMREAD_GRAYSCALE)
#     kernel = np.ones((5, 5), np.float32)/25
#     dst = cv2.filter2D(img, -1, kernel)
#     plt.xticks([]), plt.yticks([])
#     plt.imshow(dst, cmap='gray', interpolation='bicubic')
#     plt.show()
# testimg3()

filter = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
rImage = np.zeros((3, 3), np.uint8)
image = cv2.imread('D:/vsc_project/machinelearning_study/test.jpg', cv2.IMREAD_GRAYSCALE)
rImage = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_AREA)
for y in range(26):
  for x in range(26):
    rImage[x][y] = image[x][y]*filter[0][0] + image[x+1][y]*filter[0][1] + image[x+2][y]*filter[0][2] \
                 + image[x][y+1]*filter[1][0] + image[x+1][y+1]*filter[1][1] + image[x+2][y+1]*filter[1][2] \
                 + image[x][y+2]*filter[2][0] + image[x+1][y+2]*filter[2][1] + image[x+2][y+2]*filter[2][2]
plt.imshow(rImage, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()