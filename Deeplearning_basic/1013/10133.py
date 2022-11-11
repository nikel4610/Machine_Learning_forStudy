from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

df = pd.read_csv('D:/vsc_project/machinelearning_study/data-master/house_train.csv')
df = pd.get_dummies(df) # 카테고리형 변수를 0과 1로 바꾸기
df = df.fillna(df.mean()) # nan값을 평균값으로 바꾸기

# 집 값을 제외한 나머지 열을 저장합니다.
cols_train=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF', '1stFlrSF','FullBath', 'BsmtQual_Ex', 'TotRmsAbvGrd']
X_train_pre = df[cols_train]

# 집 값을 저장합니다.
y = df['SalePrice'].values

# 전체의 80%를 학습셋으로, 20%를 테스트셋으로 지정합니다.
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.25)

# 모델의 구조를 설정합니다.
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))
model.summary()

# 모델을 실행합니다.
model.compile(optimizer ='adam', loss = 'mean_squared_error')

# 20회 이상 결과가 향상되지 않으면 자동으로 중단되게끔 합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 이름을 정합니다.
modelpath="D:/vsc_project/machinelearning_study/data-master/model/Ch15-house.hdf5"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# 실행 관련 설정을 하는 부분입니다. 전체의 20%를 검증셋으로 설정합니다.
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])

# Epoch 102/2000
# 28/28 [==============================] - 0s 4ms/step - loss: 2364842752.0000 - val_loss: 1770730112.0000

# 실제 값과 모델이 예측한 값을 한 그래프에 표시해서 비교하기
x_num = np.arange(len(y_test))
pred_prices = model.predict(X_test)
plt.figure(figsize=(10, 5))
plt.plot(x_num, y_test, label='actual')
plt.plot(x_num, pred_prices, label='prediction')
plt.legend()
plt.show()

# 예측 값과 실제 값, 실행 번호가 들어갈 빈 리스트를 만듭니다.
real_prices =[]
pred_prices = []
X_num = []
