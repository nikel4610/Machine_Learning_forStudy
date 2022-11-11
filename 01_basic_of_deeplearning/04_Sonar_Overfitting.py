import pandas as pd
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('D:/python_project/deeplearning/dataset/sonar.csv', header=None)

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

# 학습셋과 테스트셋 구분
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=200, batch_size=5)

# 테스트셋에 모델 적용
print("\n Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# ...
# Epoch 200/200
# 29/29 [==============================] - 0s 498us/step - loss: 2.2897e-04 - accuracy: 1.0000
# 2/2 [==============================] - 0s 1ms/step - loss: 0.1444 - accuracy: 0.8571
# 
#  Accuracy: 0.8571
