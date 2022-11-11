from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("D:/python_project/deeplearning/dataset/housing.csv", delim_whitespace=True, header=None)

dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=500, batch_size=30)

# 예측 값과 실제 값 비교
Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예측가격: {:.3f}".format(label, prediction))

# Epoch 500/500
# 12/12 [==============================] - 0s 545us/step - loss: 18.7088
# 실제가격: 22.600, 예측가격: 22.948
# 실제가격: 50.000, 예측가격: 28.017
# 실제가격: 23.000, 예측가격: 30.729
# 실제가격: 8.300, 예측가격: 10.479
# 실제가격: 21.200, 예측가격: 22.086
# 실제가격: 19.900, 예측가격: 21.279
# 실제가격: 20.600, 예측가격: 20.591
# 실제가격: 18.700, 예측가격: 24.655
# 실제가격: 16.100, 예측가격: 19.492
# 실제가격: 18.600, 예측가격: 11.287
