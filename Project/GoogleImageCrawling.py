import pandas as pd
from msedge.selenium_tools import Edge, EdgeOptions
import time
import os
import urllib.request

def thf1():
    options = EdgeOptions()
    options.use_chromium = True
    options.add_experimental_option("prefs", {
        "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Images",
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
    driver.get('https://www.google.co.kr/imghp?hl=ko')

    ingre = pd.read_csv('D:/vsc_project/machinelearning_study/Project/searchData/Data/ingredients1.csv', encoding='cp949')
    ingre = ingre['file'].tolist()
    # print(ingre)

    def scroll():
        """스크롤을 내리는 함수"""
        SCROLL_PAUSE_TIME = 1
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)
            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    driver.find_element_by_css_selector(".mye4qd").click()
                except:
                    break
            last_height = new_height

    for i in range(78, 80):
        driver.find_element_by_class_name('gLFyf').send_keys(ingre[i])
        driver.find_element_by_class_name('Tg7LZd').click()
        scroll()

        images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
        count = 0
        link = []
        for image in images:
            if image.get_attribute('src') is not None:
                link.append(image.get_attribute('src'))
                count += 1
            if count == 100:
                break

        for k, j in enumerate(link):
            url = j
            path = 'D:/vsc_project/machinelearning_study/Project/searchData/Images/' + ingre[i] + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            urllib.request.urlretrieve(url, path + str(k) + '.jpg')
        link = []
        driver.back()

thf1()