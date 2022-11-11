from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정
np.random.seed(3)
tf.random.set_seed(3)

df_pre = pd.read_csv("D:/python_project/deeplearning/dataset/wine.csv", header=None)
df = df_pre.sample(frac=0.15)
dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 저장 폴더 만들기
# MODEL_DIR = './model'
# if not os.path.exists(MODEL_DIR):
#     os.mkdir(MODEL_DIR)
# modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"

# 모델 업데이트 및 저장
# checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 모델 실행
history = model.fit(X, Y, validation_split=0.33, epochs=3500, batch_size=500)

# 실험 결과 오차값 저장
y_vloss = history.history['val_loss']

# 정확도 값 저장
y_acc = history.history['accuracy']

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
x_len = np.arange(len(y_acc))
plt.plot(x_len, y_acc, c="blue", label="accuracy")
plt.plot(x_len, y_vloss, c="red", label="val_loss")

plt.show()

# Epoch 3500/3500
# 2/2 [==============================] - 0s 11ms/step - loss: 0.0112 - accuracy: 0.9969 
# - val_loss: 0.1607 - val_accuracy: 0.9783
