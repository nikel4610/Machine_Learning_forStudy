# 다중 분류 문제 해결
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('D:/python_project/deeplearning/dataset/iris.csv',
                 names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# 데이터전체를 한번에 보는 그래프
# sns.pairplot(df, hue = 'class')
# plt.show()

# seed값 설정
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 분류
dataset = df.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4]

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = tf.keras.utils.to_categorical(Y)

# 모델 설정
model = Sequential()
model.add(Dense(16, input_dim = 4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# 모델 실행
model.fit(X, Y_encoded, epochs = 100, batch_size = 5)

# 결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y_encoded)[1]))

# ...
# Epoch 100/100
# 30/30 [==============================] - 0s 481us/step - loss: 0.1232 - accuracy: 0.9733
# 5/5 [==============================] - 0s 748us/step - loss: 0.1213 - accuracy: 0.9800
# 
#  Accuracy: 0.9800
