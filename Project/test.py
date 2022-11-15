from selenium import webdriver as wd
import time
import pandas as pd

driver = wd.Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe')
driver.get('https://datalab.naver.com/keyword/trendSearch.naver')

df = pd.read_csv('D:/vsc_project/machinelearning_study/Project/RCPTitle3.csv')
# print(df.head())

df_dict = df.to_dict()
df_dict = list(df_dict['RCP_NM'].values())

for i in range(len(df_dict)):
    # 키워드
    driver.find_element_by_id('item_keyword1').send_keys(df_dict[i])
    # 시작 년도 월 일
    driver.find_element_by_id('startYear').click()
    driver.find_element_by_xpath('//*[@id="startYearDiv"]/ul/li[6]/a').click()
    driver.find_element_by_id('startMonth').click()
    driver.find_element_by_xpath('//*[@id="startMonthDiv"]/ul/li[1]/a').click()
    driver.find_element_by_id('startDay').click()
    driver.find_element_by_xpath('//*[@id="startDayDiv"]/ul/li[1]/a').click()

    # 종료 년도 월 일
    driver.find_element_by_id('endYear').click()
    driver.find_element_by_xpath('//*[@id="endYearDiv"]/ul/li[6]/a').click()
    driver.find_element_by_id('endMonth').click()
    driver.find_element_by_xpath('//*[@id="endMonthDiv"]/ul/li[12]/a').click()
    driver.find_element_by_id('endDay').click()
    driver.find_element_by_xpath('//*[@id="endDayDiv"]/ul/li[31]/a').click()

    if i == 0:
        # 여성 / 남성 -> tiem_gender_2
        driver.find_element_by_id('item_gender_2').click()

        # 10대
        driver.find_element_by_id('item_age_11').click()

    # 검색
    driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/a/span').click()
    time.sleep(2)

    # 다운로드 후 뒤로가기
    driver.find_element_by_xpath('//*[@id="content"]/div[1]/div[1]/div[1]/div/div/div/div/div/div[1]/div[4]/a').click()
    time.sleep(2)
    driver.back()

    # 검색어 지우기
    driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[1]/button/span').click()
    driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[2]/button/span').click()
    time.sleep(2)
