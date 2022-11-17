from selenium import webdriver as wd
import time
import pandas as pd
from msedge.selenium_tools import Edge, EdgeOptions

# TODO 자기전에 셀레니움 돌리고 자기
options = EdgeOptions()
options.use_chromium = True
options.add_experimental_option("prefs", {
    "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Keyword_Count",
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})
driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
driver.get('https://keywordsound.com/service/keyword-analysis')

df = pd.read_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data/RCP_RE_NM.csv', encoding='cp949')
# print(df.head())

df_dict = df.to_dict()
df_dict = list(df_dict['CKG_NM'].values())

for i in range(1823, len(df_dict)): # 1788, 1789
    driver.find_element_by_xpath('//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
    driver.find_element_by_xpath('//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

    driver.find_element_by_xpath('//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
    time.sleep(15)  # 15초대기

    # 날짜 선택
    driver.find_element_by_xpath('//*[@id="inputDateRange"]').click()
    # 직접 선택 클릭
    driver.find_element_by_xpath('//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

    # 다운로드 클릭
    driver.find_element_by_xpath('//*[@id="btnExportExcel"]/span').click()

    # 되돌아가기
    driver.back()