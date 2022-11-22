import pandas as pd
from multiprocessing import Pool
from selenium import webdriver as wd
from msedge.selenium_tools import Edge, EdgeOptions
import time

# options = EdgeOptions()
# options.use_chromium = True
# options.add_experimental_option("prefs", {
#     "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Images",
#     "download.prompt_for_download": False,
#     "download.directory_upgrade": True,
#     "safebrowsing.enabled": True
# })
# driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
# driver.get('https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl')

rcp = pd.read_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data/ingredients.csv', encoding='cp949')
rcp = rcp['CKG_MTRL_CN'].tolist()
rcp = [i.split('|') for i in rcp]
rcp = [i for j in rcp for i in j]

for i in range(len(rcp)):
    rcp[i] = rcp[i].strip()
    if ']' in rcp[i]:
        rcp[i] = rcp[i].split(']')[1]
    rcp[i] = rcp[i].strip()
    rcp[i] = rcp[i].split(' ')[0]

# () 안의 단어 제거
for i in range(len(rcp)):
    if '(' in rcp[i]:
        rcp[i] = rcp[i].split('(')[0]
    rcp[i] = rcp[i].strip()
    rcp[i] = rcp[i].split(' ')[0]

rcp = list(set(rcp))
rcp = [i for i in rcp if i != '']

print(rcp)
# csv로 저장
pd.DataFrame(rcp).to_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data/ingredients_2.csv', encoding='cp949', index=False)