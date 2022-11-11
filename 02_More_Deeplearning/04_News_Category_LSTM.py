# LSTM을 이용해 뉴스 카테고리 분석하기

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# seed값
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

# 불러온 데이터를 학습셋과 데이터셋으로 나누기
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

# 데이터 전처리
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 모델의 설정
model = Sequential()
model.add(Embedding(1000, 128))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test))

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))

# 테스트셋 오차
y_vloss = history.history['val_loss']

# 학습셋 오차
y_loss = history.history['loss']

# 그래프로 그리기
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

# 그래프에 그리드 추가하고 레이블 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# Epoch 20/20
# 90/90 [==============================] - 1s 11ms/step - loss: 0.7324 - accuracy: 0.8126 - val_loss: 1.2357 - val_accuracy: 0.7093
# 71/71 [==============================] - 0s 3ms/step - loss: 1.2357 - accuracy: 0.7093
# 
#  Test Accuracy: 0.7093
