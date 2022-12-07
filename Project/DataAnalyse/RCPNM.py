import pandas as pd
import os

path = 'D:/vsc_project/machinelearning_study/Project/searchData/'
df1 = pd.read_csv(os.path.join(path, 'RCP_RE.csv'), encoding='cp949')
# print(df1)

df = pd.DataFrame()
# df에 df1의 CKG_NM과 CKG_MTRL_CN을 넣는다
df['RCP_NM'] = df1['CKG_NM']
df['RCP_MTRL_CN'] = df1['CKG_MTRL_CN']

# 빈칸을 기준으로 나눈다
df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].str.split(' ')

df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].apply(lambda x: [item for item in x if '|' not in item])
df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].apply(lambda x: [item for item in x if '[' not in item])
df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].apply(lambda x: [item for item in x if ']' not in item])
df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].apply(lambda x: [item for item in x if 'or' not in item])
df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].apply(lambda x: [item for item in x if '▶' not in item])
df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].apply(lambda x: [item for item in x if ':' not in item])
df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].apply(lambda x: [item.strip() for item in x])
df['RCP_MTRL_CN'] = df['RCP_MTRL_CN'].apply(lambda x: [item for item in x if not any(c.isdigit() for c in item)])
df['Count'] = df['RCP_MTRL_CN'].apply(lambda x: len(x))
df = df[df['Count'] > 1]
df = df.sort_values(by=['Count'], axis=0, ascending=True)
df.reset_index(drop=True, inplace=True)
print(df)

# 재료 입력 받기
input_mtrl = input('재료를 입력하세요: ')
input_mtrl = input_mtrl.split(' ')

for i in range(len(df)):
    if all(mtrl in df['RCP_MTRL_CN'][i] for mtrl in input_mtrl):
        df['RCP_MTRL_CN'][i].sort(key=len)
        print(df['RCP_NM'][i], df['RCP_MTRL_CN'][i])
