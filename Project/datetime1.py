import os
import pandas as pd
from datetime import datetime

file_format = ".xlsx"
file_path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Total_Male_with_keywords'
file_name_os = os.listdir(file_path)
# # print(file_name_os)

file_list = [f"{file_path}/{file}" for file in os.listdir(file_path) if file_format in file]
# print(file_list)
for i in range(len(file_list)):
    M_10 = pd.read_excel(file_list[i])
    M_10['날짜'] = pd.date_range(start='2021-01-01', end='2021-12-31')

    if M_10['날짜'][0].month == 12 or M_10['날짜'][0].month == 1 or M_10['날짜'][0].month == 2:
        M_10['분기'] = 4
    elif M_10['날짜'][0].month == 3 or M_10['날짜'][0].month == 4 or M_10['날짜'][0].month == 5:
        M_10['분기'] = 1
    elif M_10['날짜'][0].month == 6 or M_10['날짜'][0].month == 7 or M_10['날짜'][0].month == 8:
        M_10['분기'] = 2
    elif M_10['날짜'][0].month == 9 or M_10['날짜'][0].month == 10 or M_10['날짜'][0].month == 11:
        M_10['분기'] = 3
