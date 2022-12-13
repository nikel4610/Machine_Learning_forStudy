# import splitfolders
#
# splitfolders.ratio('D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish',
#                       seed = 1337, ratio = (0.8, 0.1, 0.1),
#                       output = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_getSplit')

import os
# path = 'D:/vsc_project/machinelearning_study/Project/searchData/Ingredient_Finish_getSplit'
#
# train_dir = os.path.join(path, 'train')
# # train_dir의 모든 파일명을 리스트로 저장
# train_dir_list = os.listdir(train_dir)
# # print(train_dir_list)
#
# # train_dir_list를 txt파일로 저장
# with open(os.path.join(path, 'train_dir_list.txt'), 'w') as f:
#     for i in range(len(train_dir_list)):
#         f.write(train_dir_list[i] + '\n')

path = 'D:/vsc_project/machinelearning_study'

# train.txt 파일을 읽어서 리스트로 저장
with open(os.path.join(path, 'valid.txt'), 'r', encoding='utf-8') as f:
    train_list = f.readlines()
    print(train_list)

# train_list에서 /content/drive/MyDrive/yolov4/ 에서 /content/drive/MyDrive/yolov4/ 를 제거
for i in range(len(train_list)):
    train_list[i] = train_list[i].replace('/content/drive/MyDrive/yolov4/', '')
    print(train_list[i])


