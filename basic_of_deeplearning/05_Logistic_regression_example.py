import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부시간: x, 합격여부: y
data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12 ,1], [14, 1]]

x_data = [i[0] for i in data]
y_data = [i[1] for i in data]
plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)

a = 0
b = 0
lr = 0.05

#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

# 경사 하강법
for i in range(2001):
    for x_data, y_data in data:
        a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
        b_diff = sigmoid(a * x_data + b) - y_data
        a = a - lr * a_diff
        b = b - lr * b_diff
        if i % 1000 == 0:
            print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))

# 앞서 구한 기울기와 절편 이용해 그래프 그리기
plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
x_range = (np.arange(0, 15, 0.1)) # x값 범위
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a * x + b) for x in x_range]))
plt.show()

# epoch=0, 기울기=-0.0500, 절편=-0.0250
# epoch=0, 기울기=-0.1388, 절편=-0.0472
# epoch=0, 기울기=-0.2268, 절편=-0.0619
# epoch=0, 기울기=0.1201, 절편=-0.0185
# epoch=0, 기울기=0.2374, 절편=-0.0068
# epoch=0, 기울기=0.2705, 절편=-0.0040
# epoch=0, 기울기=0.2860, 절편=-0.0029
# epoch=1000, 기울기=1.4978, 절편=-9.9401
# epoch=1000, 기울기=1.4940, 절편=-9.9411
# epoch=1000, 기울기=1.4120, 절편=-9.9547
# epoch=1000, 기울기=1.4949, 절편=-9.9444
# epoch=1000, 기울기=1.4982, 절편=-9.9440
# epoch=1000, 기울기=1.4984, 절편=-9.9440
# epoch=1000, 기울기=1.4985, 절편=-9.9440
# epoch=2000, 기울기=1.9065, 절편=-12.9489
# epoch=2000, 기울기=1.9055, 절편=-12.9491
# epoch=2000, 기울기=1.8515, 절편=-12.9581
# epoch=2000, 기울기=1.9057, 절편=-12.9514
# epoch=2000, 기울기=1.9068, 절편=-12.9513
# epoch=2000, 기울기=1.9068, 절편=-12.9513
# epoch=2000, 기울기=1.9068, 절편=-12.9513
