from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
# ReLU를 사용하면 학습이 불안정해져 조금 변형한 LeakyReLU를 사용하는 것이 좋다.
from keras.models import Sequential, Model

import numpy as np
import matplotlib.pyplot as plt

# 생성자 모델 만들기
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2))) # 임의로 정한 노드 수: 128, 100차원 크기의 랜덤 벡터
# -> 7*7인 이유: UpSampling2D를 거쳐 14*14 , 28*28의 이미지로 변환
generator.add(BatchNormalization())
generator.add(Reshape((7, 7, 128))) # Conv2D 함수의 input_shape 부분에 들어갈 형태로 정한다
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=5, padding='same')) # padding = 'same' 조건으로 모자란 부분은 0으로 채운도
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))

# 판별자 모델 만들기
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28, 28, 1), padding='same'))
# strides = 2 이므로 이미지를 절반으로 줄여서 작업한다
# *stride는 커널 윈도의 이동 크기
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

# 모델 연결시키는 gen 모델 만들기
ginput = Input(shape=(100,)) # 랜덤한 100개의 벡터 입력

dis_output = discriminator(generator(ginput))
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()

# 신경망을 실행시키는 함수
def gen_train(epoch, batch_size, saving_interval):
    # MNIST 데이터 불러오기
    # 이미지만 사용하므로 X_tran만 호출
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    # 28*28의 이미지와 흑백이므로 1채널을 가진다
    # 127.5를 빼준 뒤 127.5로 나눠서 -1 ~ 1사이의 값으로 바꾸기
    X_train = (X_train - 127.5) / 127.5
    true = np.ones((batch_size, 1))
    false = np.zeros((batch_size, 1))

    for i in range(epoch):
        # 실제 데이터를 판별자에 입력
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        d_loss_real = discriminator.train_on_batch(imgs, true)

        # 가상 이미지를 판별자에 입력
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, false)

        # 판별자와 생성자의 오차 개선
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = gan.train_on_batch(noise, true)

        print("epoch: %d" % i, "d_loss: %.4f" % d_loss, "g_loss: %.4f" % g_loss)

        # 중간 과정을 이미지로 저장

        if i % saving_interval == 0:
            noise = np.random.normal(0, 1, (25, 100))
            gen_imgs = generator.predict(noise)

            gen_imgs = 0.5 * gen_imgs + 0.5

            fig, axs = plt.subplots(5, 5)
            cnt = 0
            for j in range(5):
                for k in range(5):
                    axs[j, k].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                    axs[j, k].axis('off')
                    cnt += 1
            fig.savefig('gan_%d.png' % i)

# 4000번 반복되고 배치크기 32, 200번마다 결과 저장
gen_train(4001, 32, 200)

# ...
# epoch: 3998 d_loss: 0.5143 g_loss: 1.5527
# epoch: 3999 d_loss: 0.3994 g_loss: 1.5678
# epoch: 4000 d_loss: 0.4466 g_loss: 1.5710
