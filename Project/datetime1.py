import os
import pandas as pd
from datetime import datetime

file_format = ".xlsx"
file_path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Total_Female_with_keywords'
file_name_os = os.listdir(file_path)
# # print(file_name_os)

file_list = [f"{file_path}/{file}" for file in os.listdir(file_path) if file_format in file]
# print(file_list)
for i in range(len(file_list)):
    M_10 = pd.read_excel(file_list[i])
    M_10['날짜'] = pd.date_range(start='2021-01-01', end='2021-12-31')

    # M_10의 날짜를 datetime으로 변환
    M_10['날짜'] = M_10['날짜'].astype('datetime64[ns]')
    M_10.loc[(M_10['날짜'].dt.month == 12) | (M_10['날짜'].dt.month == 1) | (M_10['날짜'].dt.month == 2), '분기'] = 4
    M_10.loc[(M_10['날짜'].dt.month == 3) | (M_10['날짜'].dt.month == 4) | (M_10['날짜'].dt.month == 5), '분기'] = 1
    M_10.loc[(M_10['날짜'].dt.month == 6) | (M_10['날짜'].dt.month == 7) | (M_10['날짜'].dt.month == 8), '분기'] = 2
    M_10.loc[(M_10['날짜'].dt.month == 9) | (M_10['날짜'].dt.month == 10) | (M_10['날짜'].dt.month == 11), '분기'] = 3
    M_10['분기'] = M_10['분기'].astype('int64')

    # csv로 저장
    pd.DataFrame(M_10).to_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data/Total_Female_with_keywords' + str(i) + '.csv', encoding='cp949', index=False)
