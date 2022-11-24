import pandas as pd
import os
import datetime

file_format = ".csv"

file_path_FE_Metro = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Ju/Metro/Total_Female_Proprecessing(Metro)'
file_name_os_FE_Metro = os.listdir(file_path_FE_Metro)
file_list_FE_Metro = [f"{file_path_FE_Metro}/{file}" for file in os.listdir(file_path_FE_Metro) if file_format in file]

file_path_M_Metro = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Ju/Metro/Total_Male_Proprecessing(Metro)'
file_name_os_M_Metro = os.listdir(file_path_M_Metro)
file_list_M_Metro = [f"{file_path_M_Metro}/{file}" for file in os.listdir(file_path_M_Metro) if file_format in file]

file_path_FM_Pro = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Ju/Province/Total_Female_Proprecessing(Province)'
file_name_os_FM_Pro = os.listdir(file_path_FM_Pro)
file_list_FM_Pro = [f"{file_path_FM_Pro}/{file}" for file in os.listdir(file_path_FM_Pro) if file_format in file]

file_path_M_Pro = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Ju/Province/Total_Male_Proprecessing(Province)'
file_name_os_M_Pro = os.listdir(file_path_M_Pro)
file_list_M_Pro = [f"{file_path_M_Pro}/{file}" for file in os.listdir(file_path_M_Pro) if file_format in file]

file_path_FM_Seoul = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Ju/Seoul/Total_Female_Proprecessing'
file_name_os_FM_Seoul = os.listdir(file_path_FM_Seoul)
file_list_FM_Seoul = [f"{file_path_FM_Seoul}/{file}" for file in os.listdir(file_path_FM_Seoul) if file_format in file]

file_path_M_Seoul = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/Ju/Seoul/Total_Male_Proprecessing'
file_name_os_M_Seoul = os.listdir(file_path_M_Seoul)
file_list_M_Seoul = [f"{file_path_M_Seoul}/{file}" for file in os.listdir(file_path_M_Seoul) if file_format in file]

