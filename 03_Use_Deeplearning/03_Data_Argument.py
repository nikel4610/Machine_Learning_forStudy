# 치매환자의 뇌인지 일반인의 뇌인지 예측하기
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics

np.random.seed(3)
tf.random.set_seed(3)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True, # 수평 대칭 이미지 50% 확률로 생성
                                   width_shift_range=0.1, # 전체 크기의 10% 범위에서 좌우 이동
                                   height_shift_range=0.1, # 전체 크기의 10% 범위에서 상하 이동
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'D:/python_project/deeplearning/run_project/train',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary')

# 테스트셋은 이미지 그대로
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'D:/python_project/deeplearning/run_project/test',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary')

# CNN 모델 만들어 적용
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.optimizers.Adam(learning_rate=0.0002),
              metrics=['accuracy'])

# 모델 실행
history = model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=20,
    validation_data=test_generator,
    validation_steps=4)

# 결과를 그래프로 그리기
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker='.', c='red', label='Trainset_acc')
plt.plot(x_len, val_acc, marker='.', c='lightcoral', label='Testset_acc')
plt.plot(x_len, y_vloss, marker='.', c='cornflowerblue', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss and accuracy')
plt.show()

# ...
# Epoch 20/20
# 30/30 [==============================] - 1s 17ms/step - loss: 0.1556 - accuracy: 0.9467 - val_loss: 0.0395 - val_accuracy: 1.0000
