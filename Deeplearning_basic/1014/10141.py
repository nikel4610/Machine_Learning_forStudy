from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

df = pd.read_csv('D:/vsc_project/machinelearning_study/archive/BTC-2021min.csv')

# print(df.isnull().sum().sort_values(ascending=False))

# df의 unix, date, symbol 제거
df = df.drop(['unix', 'symbol', 'Volume BTC', 'Volume USD'], axis=1)
# print(df.head())
df['price mean'] = df[['open', 'high', 'low', 'close']].mean(axis = 1)
df['spread'] = df['high'] - df['low']
df['trade'] = df['close'] - df['open']
df['buy/sell'] = df['close'].diff(periods=1)
df['buy/sell'] = df['buy/sell'].apply(lambda x: 0 if x<=0 else 1)
df['date'] = df['date'].astype('datetime64[ns]')

cols_train = ['open', 'high', 'low', 'close', 'price mean', 'spread', 'trade']
df_train = df[cols_train]

df = df.set_index('date')
(df.index[1:] - df.index[:-1]).value_counts()

X_train_pre = df_train.values
y_train_pre = df['buy/sell'].values

X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y_train_pre, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(64, input_dim=7, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15)

modelpath = 'D:/vsc_project/machinelearning_study/data-master/model/coin_model.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])

# 테스트 정확도 출력
print('정확도 : %.4f' % (model.evaluate(X_test, y_test)[1]))
# 정확도 : 0.4920
# 3818/3818 [==============================] - 3s 897us/st
