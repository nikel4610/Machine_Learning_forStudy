import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 시간 리스트
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 그래프로 그리기
plt.figure(figsize = (8,5))
plt.scatter(x, y)
plt.show()

# x y값을 넘파이 배열로 바꾸기
# 인덱스를 주어 하나씩 불러 계산이 가능하게 하기 위함
x_data = np.array(x)
y_data = np.array(y)

# 기울기a 절편b초기화
a = 0
b = 0

# 학습률 정하기
lr = 0.03

# 에폭시
epochs = 2001

# 경사 하강법
for i in range(epochs):
    y_pred = a * x_data + b # y구하기
    error = y_data - y_pred # 오차 구하기
    # 오차 함수를 a, b로 미분
    a_diff = -(2/len(x_data)) * sum(x_data * (error))
    b_diff = -(2/len(x_data)) * sum(error)

    # 학습률을 곱해 a, b 값 업데이트
    a = a - lr * a_diff
    b = b - lr * b_diff

    # 100번 반복할 때 마다 a, b값 출력
    if i % 100 == 0:
        print("epoch = %.f, 기울기 = %.04f, 절편= %.04f" % (i, a, b))

# 앞서 구한 기울기와 절편 이용해 그래프 다시 그리기
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()

# epoch = 0, 기울기 = 27.8400, 절편= 5.4300
# epoch = 100, 기울기 = 7.0739, 절편= 50.5117
# epoch = 200, 기울기 = 4.0960, 절편= 68.2822
# epoch = 300, 기울기 = 2.9757, 절편= 74.9678
# epoch = 400, 기울기 = 2.5542, 절편= 77.4830
# epoch = 500, 기울기 = 2.3956, 절편= 78.4293
# epoch = 600, 기울기 = 2.3360, 절편= 78.7853
# epoch = 700, 기울기 = 2.3135, 절편= 78.9192
# epoch = 800, 기울기 = 2.3051, 절편= 78.9696
# epoch = 900, 기울기 = 2.3019, 절편= 78.9886
# epoch = 1000, 기울기 = 2.3007, 절편= 78.9957
# epoch = 1100, 기울기 = 2.3003, 절편= 78.9984
# epoch = 1200, 기울기 = 2.3001, 절편= 78.9994
# epoch = 1300, 기울기 = 2.3000, 절편= 78.9998
# epoch = 1400, 기울기 = 2.3000, 절편= 78.9999
# epoch = 1500, 기울기 = 2.3000, 절편= 79.0000
# epoch = 1600, 기울기 = 2.3000, 절편= 79.0000
# epoch = 1700, 기울기 = 2.3000, 절편= 79.0000
# epoch = 1800, 기울기 = 2.3000, 절편= 79.0000
# epoch = 1900, 기울기 = 2.3000, 절편= 79.0000
# epoch = 2000, 기울기 = 2.3000, 절편= 79.0000
