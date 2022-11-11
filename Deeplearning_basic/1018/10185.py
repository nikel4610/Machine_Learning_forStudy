from selenium import webdriver as wd
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service

s = Service(executable_path='D:/vsc_project/machinelearning_study/msedgedriver.exe')
driver = wd.Edge(service=s)
url = 'https://www.coupang.com/'
driver.get(url)
driver.implicitly_wait(10)

keyword = '삼겹살'
element = driver.find_element(By.ID, 'headerSearchKeyword')
element.send_keys(keyword)
element.submit()
driver.implicitly_wait(10)