import pandas as pd
import os

path = 'D:/vsc_project/machinelearning_study/Project/searchData'
df1 = pd.read_csv(os.path.join(path, 'Last_all.csv'), encoding='cp949')
df2 = pd.read_csv(os.path.join(path, 'RCP_RE1.csv'), encoding='cp949')

# df1 = df1[['CKG_NM', 'CKG_MTRL_CN']]

df1 = df1[['RCP_N']]
df1 = df1.drop_duplicates(['RCP_N'], keep='first')
df1 = df1.reset_index(drop=True)

# df2의 CKG_NM과 df1의 RCP_N이 같은 행 출력
df3 = pd.merge(df1, df2, left_on='RCP_N', right_on='CKG_NM', how='inner')
df3 = df3[['CKG_NM', 'CKG_MTRL_CN']]
print(df3)

# csv로 저장
df3.to_csv(os.path.join(path, 'rcp.csv'), encoding='cp949', index=False)