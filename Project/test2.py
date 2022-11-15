from selenium import webdriver as wd
import time
import pandas as pd

driver = wd.Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe')
driver.get('https://keywordsound.com/service/keyword-analysis')

df = pd.read_csv('D:/vsc_project/machinelearning_study/Project/RCPTitle3.csv')
# print(df.head())

df_dict = df.to_dict()
df_dict = list(df_dict['RCP_NM'].values())

for i in range(len(df_dict)):
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