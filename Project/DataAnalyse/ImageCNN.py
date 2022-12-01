import os
import pandas as pd
from PIL import Image as img
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

file_path = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish'
file_name_os = os.listdir(file_path)
# print(file_name_os)

img_list = []
for i in range(len(file_name_os)):
    path = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish' + '/' + file_name_os[i]
    file_list = [f"{path}/{file}" for file in os.listdir(path) if '.jpg' in file]
    img_list.append(file_list)
    # print(file_list)

# print(img_list)

# # 이미지의 절대 경로를 txt 파일로 저장
# with open('D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_getSplit/test.txt', 'w') as f:
#     for i in img_list:
#         for j in i:
#             f.write(j)


# img_list의 모든 이미지 크기를 통일
for i in range(len(img_list)):
    for j in range(len(img_list[i])):
        image = img.open(img_list[i][j])
        image = image.convert('RGB')
        image = image.resize((512, 512))
        image.save(img_list[i][j])

# 이미지 생성 인자
data_gen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.5,
    zoom_range = [0.8, 2.0],
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'constant'
)

# 이미지 추가 생성
for i in range(len(img_list)):
    for j in range(len(img_list[i])):
        image = tf.keras.utils.load_img(img_list[i][j])
        save_dir = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_gen' + '/' + file_name_os[i]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        x = tf.keras.utils.img_to_array(image)
        x = x.reshape((1, ) + x.shape)

        g = data_gen.flow(x, batch_size = 1, save_to_dir = save_dir, save_prefix = file_name_os[i], save_format = 'jpg')

        for k in range(30):
            g.next()