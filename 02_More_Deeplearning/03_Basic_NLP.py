# 영화 리뷰가 긍정적인지 부정적인지 예측하기

import numpy as np
import tensorflow as tf
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

# 텍스트 리뷰 자료 지정
docs = ['너무 재밌어요', '최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요',
        '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정 리뷰는 1 부정 리뷰는 0으로 클래스 지정
classes = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

# 토큰에 지정된 인덱스로 새로운 배열 생성
x = token.texts_to_sequences(docs)

# 서로 다른 길이의 데이터를 4개로 맞추기 (패딩)
padded_x = pad_sequences(x, 4)
"\n패딩 결과\n", print(padded_x)

# 임베딩에 입력될 단어 수 지정
word_size = len(token.word_index) + 1

# 단어 임베딩을 포함하여 모델 생성 + 결과 출력
model = Sequential()
model.add(Embedding(word_size, 8, input_length=4))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_x, classes, epochs=20)

print("\n Accuracy: %.4f" % (model.evaluate(padded_x, classes)[1]))

# Epoch 20/20
# 1/1 [==============================] - 0s 3ms/step - loss: 0.6394 - accuracy: 0.8333
# 1/1 [==============================] - 0s 72ms/step - loss: 0.6370 - accuracy: 0.8333
# 
#  Accuracy: 0.8333
