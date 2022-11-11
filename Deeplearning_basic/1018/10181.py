from selenium import webdriver as wd
from selenium.webdriver.common.by import By

driver = wd.Edge(executable_path='D:/vsc_project/machinelearning_study/msedgedriver.exe')
url = 'http://tour.interpark.com'
driver.get(url)
driver.implicitly_wait(10)

element = driver.find_element(By.ID, 'SearchGNBText')
keyword = '도쿄'
element.send_keys(keyword)

driver.find_element(By.CLASS_NAME, 'search-btn').click()
css_selector = '#app > div > div:nth-child(1) > div.resultAtc > div.sortTabZone > div > ul > li:nth-child(4)'
driver.find_element(By.CSS_SELECTOR, css_selector).click()

ul_tag = driver.find_element(By.ID, 'boxList')
h5_tags = ul_tag.find_elements(By.CLASS_NAME, 'infoTitle')
p_tags = ul_tag.find_elements(By.CLASS_NAME, 'final')

for h5_tag in h5_tags:
    print(h5_tag.text)

for p_tag in p_tags:
    strong_tag = p_tag.find_element(By.TAG_NAME, 'strong')
    print(strong_tag.text)

page2_element = driver.find_element(By.CSS_SELECTOR, '#app > div > div:nth-child(1) > div.resultAtc > div.contentsZone > div.panelZone > div.pageNumBox > ul > li:nth-child(2)')
page2_element.click()
