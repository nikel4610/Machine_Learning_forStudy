import pandas as pd
import os
import openpyxl as xls
import glob

file_format = ".xlsx"
file_path = 'D:/vsc_project/machinelearning_study/Project/searchData/Data/mergedMaleFemale'
file_name_os = os.listdir(file_path)
# # print(file_name_os)

file_list = [f"{file_path}/{file}" for file in os.listdir(file_path) if file_format in file]
# print(file_list)

# for i in range(2, len(file_name_os)-1):
#     wb = xls.load_workbook(file_path + '/' + 'datalab' + ' ' + '(' + str(i+1) +  ')' + '.xlsx')
#     ws = wb.active
#     # ws.delete_rows(1, 6)
#     ws.delete_cols(1)
#     wb.save(file_path + '/' + 'datalab' + ' ' + '(' + str(i+1) +  ')' + '.xlsx')

# df = pd.DataFrame()
# file = file_list[-1]
#
# for i in range(1, len(file_name_os)-1):
#     df = pd.concat([df, pd.read_excel(file_list[i])], axis = 1)
# df.to_excel(file_path + '/' + 'datalabAll' + '.xlsx')

