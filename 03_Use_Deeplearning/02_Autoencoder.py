from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터셋 호출
(x_train, _), (x_test, _) = mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# 생성자 모델 만들기
autoencoder = Sequential()

# 인코딩 부분
# 입력된 값을 축소시키는 부분
autoencoder.add(Conv2D(16, kernel_size=3, padding='same', input_shape=(28, 28, 1), activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=2, padding='same'))
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu', strides=2))

# 디코딩 부분
# 다시 차원을 점차 늘려 입력된 값과 똑같은 크기의 출력 값을 내보내는 부분
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(8, kernel_size=3, activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(1, kernel_size=3, activation='sigmoid', padding='same'))

# 전체 구조 확인
autoencoder.summary()

# 컴파일 및 학습
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, validation_data=(X_test, X_test))

# 학습된 결과를 출력
test = np.random.randint(X_test.shape[0], size = 5)

# 테스트할 이미지 랜덤 출력
# 앞서 만든 오토인코더 모델에 넣기
ae_imgs = autoencoder.predict(X_test)
# 출력 이미지 크기 정하기
plt.figure(figsize=(7, 2))

for i, image_idx in enumerate(test):
    # 랜덤으로 뽑은 이미지 나열
    ax = plt.subplot(2, 7, i + 1)
    # 테스트 할 이미지 그대로 보여주기
    plt.imshow(X_test[image_idx].reshape(28, 28))
    ax.axis('off')
    ax = plt.subplot(2, 7, 7 + i + 1)
    # 오토인코딩 결과를 다음 열에 출력
    plt.imshow(ae_imgs[image_idx].reshape(28, 28))
    ax.axis('off')
plt.show()

# ...
# Epoch 48/50
# 469/469 [==============================] - 3s 6ms/step - loss: 0.0844 - val_loss: 0.0838
# Epoch 49/50
# 469/469 [==============================] - 3s 6ms/step - loss: 0.0844 - val_loss: 0.0832
# Epoch 50/50
# 469/469 [==============================] - 3s 6ms/step - loss: 0.0842 - val_loss: 0.0830
