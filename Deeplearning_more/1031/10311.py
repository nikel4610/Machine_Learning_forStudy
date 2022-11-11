import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# DNN을 통한 MNIST 분류
fs = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fs.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()
# model.add(layers.Flatten(input_shape=(28, 28)))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])
#
# model.fit(train_images, train_labels, epochs=10)
# ...
# Epoch 9/10
# 1875/1875 [==============================] - 6s 3ms/step - loss: 0.2566 - accuracy: 0.9031
# Epoch 10/10
# 1875/1875 [==============================] - 6s 3ms/step - loss: 0.2491 - accuracy: 0.9056

# CNN을 통한 MNIST 분류
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
# ...
# Epoch 9/10
# 1875/1875 [==============================] - 6s 3ms/step - loss: 0.1613 - accuracy: 0.9394
# Epoch 10/10
# 1875/1875 [==============================] - 6s 3ms/step - loss: 0.1472 - accuracy: 0.9432
# 313/313 [==============================] - 1s 2ms/step - loss: 0.2830 - accuracy: 0.9054