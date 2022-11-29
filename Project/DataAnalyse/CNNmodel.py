from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping

import numpy as np
import os
import splitfolders
import tensorflow as tf
import matplotlib.pyplot as plt

# splitfolders.ratio('D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_gen',
#                       seed = 1337, ratio = (0.8, 0.1, 0.1),
#                       output = 'D:/vsc_project/machinelearning_study/Project/searchData/splitData')

base_dir = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_getSplit'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# 경로 내 파일
train_dir_list = os.listdir(train_dir)
validation_dir_list = os.listdir(validation_dir)
test_dir_list = os.listdir(test_dir)

# 파일 내 이미지 파일
for i in range(len(train_dir_list)):
    train_dir_fnames = os.listdir(os.path.join(train_dir, train_dir_list[i]))
    validation_dir_fnames = os.listdir(os.path.join(validation_dir, validation_dir_list[i]))
    test_dir_fnames = os.listdir(os.path.join(test_dir, test_dir_list[i]))

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

early_stopping = EarlyStopping(monitor='val_loss', patience=13)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=30,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                    target_size=(150, 150),
                                                    batch_size=30,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                    target_size=(150, 150),
                                                    batch_size=30,
                                                    class_mode='categorical')

# 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(206, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'])

history = model.fit_generator(train_generator,
                                steps_per_epoch=100,
                                epochs=300,
                                validation_data=validation_generator,
                                validation_steps=50,
                                callbacks=[early_stopping])

model.save('D:/vsc_project/machinelearning_study/Project/searchData/ingredient_model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','val'])
plt.show()

# 모델 평가
print("-- Evaluate --")
scores = model.evaluate(test_generator)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
# acc: 73.29%