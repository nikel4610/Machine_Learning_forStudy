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

# 학습 자동 중단 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=100)

# 모델 실행
model.fit(X, Y, validation_split=0.2, epochs=2000, batch_size=500, callbacks=[early_stopping])

print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

# Epoch 1197/2000
# 2/2 [==============================] - 0s 13ms/step - loss: 0.0349 - accuracy: 0.9885 - val_loss: 0.0586 - val_accuracy: 0.9897
# 31/31 [==============================] - 0s 400us/step - loss: 0.0432 - accuracy: 0.9867
# 
#  Accuracy: 0.9867
