import pandas as pd
from multiprocessing import Pool
from selenium import webdriver as wd
from msedge.selenium_tools import Edge, EdgeOptions
import time

df = pd.read_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data/RCP_RE_NM.csv', encoding='cp949')
# print(df)

def fm_search1():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Female\FM10",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://datalab.naver.com/keyword/trendSearch.naver')
    for i in range(len(df)):
        # 키워드
        driver.find_element_by_id('item_keyword1').send_keys(df['CKG_NM'][i])
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
            driver.find_element_by_id('item_gender_1').click()

            # 나이 선택
            driver.find_element_by_id('item_age_1').click()
            driver.find_element_by_id('item_age_2').click()

        # 검색
        driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/a/span').click()
        time.sleep(2)

        # 다운로드 후 뒤로가기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[1]/div[1]/div[1]/div/div/div/div/div/div[1]/div[4]/a').click()
        time.sleep(2)
        driver.back()

        # 검색어 지우기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[1]/button/span').click()
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[2]/button/span').click()
        time.sleep(2)

def fm_search2():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Female\FM20",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://datalab.naver.com/keyword/trendSearch.naver')
    for i in range(len(df)):
        # 키워드
        driver.find_element_by_id('item_keyword1').send_keys(df['CKG_NM'][i])
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
            driver.find_element_by_id('item_gender_1').click()

            # 나이 선택
            driver.find_element_by_id('item_age_3').click()
            driver.find_element_by_id('item_age_4').click()

        # 검색
        driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/a/span').click()
        time.sleep(2)

        # 다운로드 후 뒤로가기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[1]/div[1]/div[1]/div/div/div/div/div/div[1]/div[4]/a').click()
        time.sleep(2)
        driver.back()

        # 검색어 지우기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[1]/button/span').click()
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[2]/button/span').click()
        time.sleep(2)

def fm_search3():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Female\FM30",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://datalab.naver.com/keyword/trendSearch.naver')
    for i in range(len(df)):
        # 키워드
        driver.find_element_by_id('item_keyword1').send_keys(df['CKG_NM'][i])
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
            driver.find_element_by_id('item_gender_1').click()

            # 나이 선택
            driver.find_element_by_id('item_age_5').click()
            driver.find_element_by_id('item_age_6').click()

        # 검색
        driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/a/span').click()
        time.sleep(2)

        # 다운로드 후 뒤로가기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[1]/div[1]/div[1]/div/div/div/div/div/div[1]/div[4]/a').click()
        time.sleep(2)
        driver.back()

        # 검색어 지우기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[1]/button/span').click()
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[2]/button/span').click()
        time.sleep(2)

def fm_search4():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Female\FM40",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://datalab.naver.com/keyword/trendSearch.naver')
    for i in range(len(df)):
        # 키워드
        driver.find_element_by_id('item_keyword1').send_keys(df['CKG_NM'][i])
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
            driver.find_element_by_id('item_gender_1').click()

            # 나이 선택
            driver.find_element_by_id('item_age_7').click()
            driver.find_element_by_id('item_age_8').click()

        # 검색
        driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/a/span').click()
        time.sleep(2)

        # 다운로드 후 뒤로가기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[1]/div[1]/div[1]/div/div/div/div/div/div[1]/div[4]/a').click()
        time.sleep(2)
        driver.back()

        # 검색어 지우기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[1]/button/span').click()
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[2]/button/span').click()
        time.sleep(2)

def fm_search5():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Female\FM50",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://datalab.naver.com/keyword/trendSearch.naver')
    for i in range(len(df)):
        # 키워드
        driver.find_element_by_id('item_keyword1').send_keys(df['CKG_NM'][i])
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
            driver.find_element_by_id('item_gender_1').click()

            # 나이 선택
            driver.find_element_by_id('item_age_9').click()
            driver.find_element_by_id('item_age_10').click()

        # 검색
        driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/a/span').click()
        time.sleep(2)

        # 다운로드 후 뒤로가기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[1]/div[1]/div[1]/div/div/div/div/div/div[1]/div[4]/a').click()
        time.sleep(2)
        driver.back()

        # 검색어 지우기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[1]/button/span').click()
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[2]/button/span').click()
        time.sleep(2)

def fm_search6():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Female\FM60",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://datalab.naver.com/keyword/trendSearch.naver')
    for i in range(len(df)):
        # 키워드
        driver.find_element_by_id('item_keyword1').send_keys(df['CKG_NM'][i])
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
            driver.find_element_by_id('item_gender_1').click()

            # 나이 선택
            driver.find_element_by_id('item_age_11').click()

        # 검색
        driver.find_element_by_xpath('//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/a/span').click()
        time.sleep(2)

        # 다운로드 후 뒤로가기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div[1]/div[1]/div[1]/div/div/div/div/div/div[1]/div[4]/a').click()
        time.sleep(2)
        driver.back()

        # 검색어 지우기
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[1]/button/span').click()
        driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div[1]/div/form/fieldset/div/div[1]/div[2]/button/span').click()
        time.sleep(2)