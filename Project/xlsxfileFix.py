import openpyxl as xls
import pandas as pd

# df1 = pd.read_excel('D:/vsc_project/machinelearning_study/Project/searchData/Data/datalabAll.xlsx')
# df2 = pd.read_excel('D:/vsc_project/machinelearning_study/Project/searchData/Data/datalabAll2.xlsx')
#
# # print(df1.head())
#
# df1 = df1.drop(['Unnamed: 0'], axis = 1)
# df2 = df2.drop(['Unnamed: 0'], axis = 1)
#
# df1 = df1.loc[:, (df1 != 0).any(axis=0)]
# df2 = df2.loc[:, (df2 != 0).any(axis=0)]
#
# df = pd.concat([df1, df2], axis = 1)
# df.to_excel('D:/vsc_project/machinelearning_study/Project/searchData/Data/datalabAll3.xlsx')