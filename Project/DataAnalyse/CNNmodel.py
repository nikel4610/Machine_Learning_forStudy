from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D

import numpy as np
import splitfolders

# splitfolders.ratio('D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_gen',
#                       seed = 1337, ratio = (0.8, 0.1, 0.1),
#                       output = 'D:/vsc_project/machinelearning_study/Project/searchData/splitData')

train_data_dir = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_getSplit/train'
validation_data_dir = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_getSplit/val'
test_data_dir = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_getSplit/test'
