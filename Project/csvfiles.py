import pandas as pd
import re

df = pd.read_csv('D:/vsc_project/machinelearning_study/Project/CookRCP.csv')
# print(df['RCP_PARTS_DTLS'])
list1 = []
list2 = []

hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
for i in range(len(df)):
    result = hangul.sub('', str(df['RCP_PARTS_DTLS'][i]))
    list1.append(result.split())

for i in range(len(list1)):
    for j in range(len(list1[i])):
        list2.append(list1[i][j])

list2 = list(set(list2))
print(list2)