from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

# 시드 난수 설정
np.random.seed(3)
tf.random.set_seed(3)

# 준비된 데이터셋 부르기
Data_set = np.loadtxt("ThoraricSurgery.csv", delimiter=",")

# 환자의 기록과 수술 결과 구분하여 설정
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

# 딥러닝 모델 설정
model = Sequential()
# 출력층 /. 노드30개, 데이터에서 받을 17개의 값으로 30개 노드로 보냄
model.add(Dense(30, input_dim=17, activation='relu'))
# 출력층 /. 노드 1개
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# 10번씩 끊어서 총 100번 실행 반복
model.fit(X, Y, epochs=100, batch_size=10)

# Epoch 100/100
# 47/47 [==============================] - 0s 4ms/step - loss: 0.1173 - accuracy: 0.8596
