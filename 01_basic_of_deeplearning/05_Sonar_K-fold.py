from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import tensorflow as tf

# seed값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv('D:/python_project/deeplearning/dataset/sonar.csv', header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 18개의 파일로 쪼갠다
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
accur = []

# 모델의 설정 컴파일 실행
for train, test in skf.split(X, Y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X[train], Y[train], epochs=100, batch_size=10)
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1])
    accur.append(k_accuracy)

print("\n %.f fold accuracy:" % n_fold, accur)

# ...
# Epoch 100/100
# 19/19 [==============================] - 0s 498us/step - loss: 0.1799 - accuracy: 0.9415
# 1/1 [==============================] - 0s 53ms/step - loss: 0.4834 - accuracy: 0.7500
# 
#  10 fold accuracy: ['0.7143', '0.8571', '0.8095', '0.8571', '0.8095',
#                     '0.8571', '0.8095', '0.7619', '0.8500', '0.7500']
