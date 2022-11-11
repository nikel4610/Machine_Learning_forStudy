from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:/vsc_project/machinelearning_study/data-master/wine.csv')
# modelpath = 'D:/vsc_project/machinelearning_study/data-master/model/all/{epoch:02d}-{val_loss:.4f}.hdf5'
# print(df.head())
# checkpointer = ModelCheckpoint(filepath=modelpath, verbose=1)

# 학습이 언제 자동 중단될지를 설정합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

X = df.iloc[:,0:12]
y = df.iloc[:,12]

# 학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조를 설정합니다.
model = Sequential()
# 모델이 3개의 은닉층을 갖도록 구조를 설계(설정)합니다.
model.add(Dense(30,  input_dim=12, activation='relu'))    # 첫 번째 은닉층은 30개 노드
model.add(Dense(12, activation='relu'))                   # 두 번째 은닉층은 12개 노드
model.add(Dense(8, activation='relu'))                    # 세 번째 은닉층은 8개 노드로 구성
model.add(Dense(1, activation='sigmoid'))                 # 출력층
model.summary()

# 모델을 컴파일합니다.
# 이진 분류를 위한 손실 함수(lost function) : binary_crossentropy
# 옵티마이저 : adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.

history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, callbacks = [early_stopping_callback]) # callbacks = [checkpointer]) # 0.8 x 0.25 = 0.8 x 1/4 = 0.2

# 테스트 결과를 출력합니다.
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])

# hist_df=pd.DataFrame(history.history)
# # y_vloss에 검증(validation) 데이터셋에 대한 오차를 저장합니다.
# y_vloss=hist_df['val_loss']
# # y_loss에 학습(train) 데이터셋의 오차를 저장합니다.
# y_loss=hist_df['loss']
#
# # x에 에포크 값을 지정하고
# # 검증 데이터셋에 대한 오차를 빨간색으로, 학습셋에 대한 오차를 파란색으로 표시합니다.
# x_len = np.arange(len(y_loss))
# plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Validation loss')
# plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Trains loss')
#
# plt.legend(loc='upper right')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
