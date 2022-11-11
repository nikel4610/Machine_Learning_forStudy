from selenium import webdriver as wd
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service

s = Service(executable_path='D:/vsc_project/machinelearning_study/msedgedriver.exe')
driver = wd.Edge(service=s)
url = 'http://www.naver.com'
driver.get(url)
driver.implicitly_wait(10)

keyword = '파이썬'
element = driver.find_element(By.ID, 'query')
element.send_keys(keyword)

css_selector = '#search_btn'
driver.find_element(By.CSS_SELECTOR, css_selector).click()

jisik_selector = '#lnb > div.lnb_group > div > ul > li:nth-child(4) > a'
driver.find_element(By.CSS_SELECTOR, jisik_selector).click()

ul_tags = driver.find_element(By.CLASS_NAME, 'lst_total._list')
Q_tags = ul_tags.find_elements(By.CLASS_NAME, 'question_group')
#main_pack > section > div > ul > li:nth-child(2) > div > div.question_area > div.question_group

for i in Q_tags:
    print(i.text, '\n')