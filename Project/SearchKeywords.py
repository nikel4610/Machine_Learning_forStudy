import pandas as pd
from multiprocessing import Pool
from selenium import webdriver as wd
from msedge.selenium_tools import Edge, EdgeOptions
import time

df = pd.read_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data/RCP_RE_NM.csv', encoding='cp949')
# print(df)
df_dict = df.to_dict()
df_dict = list(df_dict['CKG_NM'].values())
# TODO 실행 시키고 자기
def fm_search1():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Keywords",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://keywordsound.com/service/keyword-analysis')
    for i in range(1823, 7523):  # 1788, 1789
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(5)  # 15초대기

        # 날짜 선택
        driver.find_element_by_xpath('//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element_by_xpath('//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element_by_xpath('//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
def fm_search2():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Keywords2",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://keywordsound.com/service/keyword-analysis')
    for i in range(7523, 13223):  # 1788, 1789
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(5)  # 15초대기

        # 날짜 선택
        driver.find_element_by_xpath('//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element_by_xpath('//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element_by_xpath('//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
def fm_search3():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Keywords3",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://keywordsound.com/service/keyword-analysis')
    for i in range(13223, 18923):  # 1788, 1789
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(5)  # 15초대기

        # 날짜 선택
        driver.find_element_by_xpath('//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element_by_xpath('//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element_by_xpath('//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
def fm_search4():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Keywords4",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://keywordsound.com/service/keyword-analysis')
    for i in range(18923, 24623):  # 1788, 1789
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(5)  # 15초대기

        # 날짜 선택
        driver.find_element_by_xpath('//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element_by_xpath('//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element_by_xpath('//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
def fm_search5():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Keywords5",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://keywordsound.com/service/keyword-analysis')
    for i in range(24623,30323):  # 1788, 1789
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(5)  # 15초대기

        # 날짜 선택
        driver.find_element_by_xpath('//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element_by_xpath('//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element_by_xpath('//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()
def fm_search6():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Data\Keywords6",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://keywordsound.com/service/keyword-analysis')
    for i in range(30323, len(df_dict)):  # 1788, 1789
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').send_keys(df_dict[i])

        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/button').click()
        time.sleep(5)  # 15초대기

        # 날짜 선택
        driver.find_element_by_xpath('//*[@id="inputDateRange"]').click()
        # 직접 선택 클릭
        driver.find_element_by_xpath('//*[@id="kt_body"]/div[3]/div[1]/ul/li[4]').click()

        # 다운로드 클릭
        driver.find_element_by_xpath('//*[@id="btnExportExcel"]/span').click()

        # 되돌아가기
        driver.back()
        driver.find_element_by_xpath(
                            '//*[@id="kt_content_container"]/div/div/div[1]/div[2]/div/div/div/div/input').clear()