import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

df = pd.read_csv('D:/python_project/deeplearning/dataset/pima-indians-diabetes.csv',
                 names = ['pregnant', 'plasma', 'pressure', 'thickness', 'insulin', 'BMI', 'pedigree', 'age', 'class'])

# 5줄 정도 데이터를 불러오기
# print(df.head(5))
# 데이터 정보 형식 불러오기
# print(df.info())
# 좀 더 자세히 알아보기
# print(df.describe())
# 데이터의 일부만 불러오기 (임신 횟수와 발병여부)
# print(df[['pregnant', 'class']])

# 데이터 전처리
# groupby -> pregnant 정보를 기준으로 새 그룹 만들기
# 새로운 인덱스 + mean()함수로 평균을 구해 sort_values()로 오름차순 정리
# print(df[['pregnant', 'class']].groupby(['pregnant'], as_index=False).mean().sort_values
#       (by='pregnant', ascending=True))

# 그래프의 크기 설정
# plt.figure(figsize=(12,12))

# heatmap()함수를 통해 한눈에 패턴 확인 가능
# sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat,
#             linecolor='white', annot=True)
# plt.show()

# plasma와 class 항목만 따로 떼어내어 두 항목간의 상관관계를 확인
# grid = sns.FacetGrid(df, col='class')
# grid.map(plt.hist, 'plasma', bins=10)
# plt.show()

# seed값 생성
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 로드
dataset = np.loadtxt('D:/python_project/deeplearning/dataset/pima-indians-diabetes.csv',delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]

# 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
model.fit(X, Y, epochs=150, batch_size=10)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

# ...
# Epoch 150/150
# 77/77 [==============================] - 0s 498us/step - loss: 0.4795 - accuracy: 0.7591
# 24/24 [==============================] - 0s 563us/step - loss: 0.4656 - accuracy: 0.7708
#
#  Accuracy: 0.7708
