from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:/vsc_project/machinelearning_study/data-master/pima-indians-diabetes3.csv')

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

X = df.iloc[:,0:8]
y = df.iloc[:,8]

# k = 5
#
# kfold = KFold(n_splits=k, shuffle=True)
# acc_score = []
#
# def model_fn():
#     model = Sequential()
#     model.add(Dense(12, input_dim=8, activation='relu', name='Dense_1'))
#     model.add(Dense(8, activation='relu', name='Dense_2'))
#     model.add(Dense(1, activation='sigmoid',name='Dense_3'))
#     model.summary()
#     return model
#
#
# # K겹 교차 검증을 이용해 k번의 학습을 실행합니다.
# for train_index, test_index in kfold.split(X):  # for 문에 의해서 k번 반복합니다. spilt()에 의해 k개의 학습셋, 테스트셋으로 분리됩니다.
#     X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     model = model_fn()
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)
#
#     accuracy = model.evaluate(X_test, y_test)[1]  # 정확도를 구합니다.
#     acc_score.append(accuracy)  # 정확도 리스트에 저장합니다.
#
# # k번 실시된 정확도의 평균을 구합니다.
# avg_acc_score = sum(acc_score) / k
#
# # 결과를 출력합니다.
# print('정확도:', acc_score)
# print('정확도 평균:', avg_acc_score)
# # 정확도: [0.7207792401313782, 0.7467532753944397, 0.7792207598686218, 0.7843137383460999, 0.7254902124404907]
# # 정확도 평균: 0.7513114452362061

# 학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조를 설정합니다.
model = Sequential()
# 모델이 3개의 은닉층을 갖도록 구조를 설계(설정)합니다.
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid',name='Dense_3'))
model.summary()                # 출력층

# 모델을 컴파일합니다.
# 이진 분류를 위한 손실 함수(lost function) : binary_crossentropy
# 옵티마이저 : adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.

history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, callbacks = [early_stopping_callback]) # callbacks = [checkpointer]) # 0.8 x 0.25 = 0.8 x 1/4 = 0.2

# 테스트 결과를 출력합니다.
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])

hist_df=pd.DataFrame(history.history)
# y_vloss에 검증(validation) 데이터셋에 대한 오차를 저장합니다.
y_vloss=hist_df['val_loss']
# y_loss에 학습(train) 데이터셋의 오차를 저장합니다.
y_loss=hist_df['loss']

# ...
# Epoch 804/2000
# 1/1 [==============================] - 0s 25ms/step - loss: 0.5375 - accuracy: 0.7500 - val_loss: 0.5635 - val_accuracy: 0.7403
# 5/5 [==============================] - 0s 3ms/step - loss: 0.6161 - accuracy: 0.7273
# Test accuracy: 0.7272727489471436

# x에 에포크 값을 지정하고
# 검증 데이터셋에 대한 오차를 빨간색으로, 학습셋에 대한 오차를 파란색으로 표시합니다.
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Validation loss')
plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Trains loss')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()