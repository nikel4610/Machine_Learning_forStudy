import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 공부시간 = x, 성적 = y
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

# 그래프
ax = plt.axes(projection = '3d')
ax.set_xlabel('공부 시간')
ax.set_ylabel('과외 경험 횟수')
ax.set_zlabel('성적')
ax.dist = 11
ax.scatter(x1, x2, y)
plt.show()

# x y 값을 넘파이 배열로 바꾸기
# 인덱스로 하나씩 불러와 계산
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

# 기울기(a) 절편(b) 값 초기화
a1 = 0
a2 = 0
b = 0

# 학습률 , epochs
lr = 0.02
epochs = 2001

# 경사 하강법
for i in range(epochs):
    y_pred = a1 * x1_data + a2 * x2_data + b
    error = y_data - y_pred
    a1_diff = -(2 / len(x1_data)) * sum(x1_data * (error))
    a2_diff = -(2 / len(x2_data)) * sum(x2_data * (error))
    b_diff = -(2 / len(x1_data)) * sum(y_data - y_pred)
    a1 = a1 - lr * a1_diff
    a2 = a2 - lr * a2_diff
    b = b - lr * b_diff

    if i % 100 == 0:
        print("epoch=%.f, 기울기1 = %.04f, 기울기2 = %.04f, 절편=%.04f" % (i, a1, a2, b))

# epoch=0, 기울기1 = 18.5600, 기울기2 = 8.4500, 절편=3.6200
# epoch=100, 기울기1 = 7.2994, 기울기2 = 4.2867, 절편=38.0427
# epoch=200, 기울기1 = 4.5683, 기울기2 = 3.3451, 절편=56.7901
# epoch=300, 기울기1 = 3.1235, 기울기2 = 2.8463, 절편=66.7100
# epoch=400, 기울기1 = 2.3591, 기울기2 = 2.5823, 절편=71.9589
# epoch=500, 기울기1 = 1.9546, 기울기2 = 2.4427, 절편=74.7362
# epoch=600, 기울기1 = 1.7405, 기울기2 = 2.3688, 절편=76.2058
# epoch=700, 기울기1 = 1.6273, 기울기2 = 2.3297, 절편=76.9833
# epoch=800, 기울기1 = 1.5673, 기울기2 = 2.3090, 절편=77.3948
# epoch=900, 기울기1 = 1.5356, 기울기2 = 2.2980, 절편=77.6125
# epoch=1000, 기울기1 = 1.5189, 기울기2 = 2.2922, 절편=77.7277
# epoch=1100, 기울기1 = 1.5100, 기울기2 = 2.2892, 절편=77.7886
# epoch=1200, 기울기1 = 1.5053, 기울기2 = 2.2875, 절편=77.8209
# epoch=1300, 기울기1 = 1.5028, 기울기2 = 2.2867, 절편=77.8380
# epoch=1400, 기울기1 = 1.5015, 기울기2 = 2.2862, 절편=77.8470
# epoch=1500, 기울기1 = 1.5008, 기울기2 = 2.2860, 절편=77.8518
# epoch=1600, 기울기1 = 1.5004, 기울기2 = 2.2859, 절편=77.8543
# epoch=1700, 기울기1 = 1.5002, 기울기2 = 2.2858, 절편=77.8556
# epoch=1800, 기울기1 = 1.5001, 기울기2 = 2.2858, 절편=77.8563
# epoch=1900, 기울기1 = 1.5001, 기울기2 = 2.2857, 절편=77.8567
# epoch=2000, 기울기1 = 1.5000, 기울기2 = 2.2857, 절편=77.8569
