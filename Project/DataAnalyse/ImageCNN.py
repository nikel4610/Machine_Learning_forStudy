import os

path_dir = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/ingredients'
file_list = os.listdir(path_dir)

print(file_list)

import pandas as pd

df = pd.DataFrame()
df['file'] = file_list
df.to_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data/ingredients1.csv', index=False, encoding='cp949')