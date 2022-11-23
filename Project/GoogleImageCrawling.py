import pandas as pd
from multiprocessing import Pool
from selenium import webdriver as wd
from msedge.selenium_tools import Edge, EdgeOptions
import time

options = EdgeOptions()
options.use_chromium = True
options.add_experimental_option("prefs", {
    "download.default_directory": r"D:\vsc_project\machinelearning_study\Project\searchData\Images",
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})
driver = Edge('D:/vsc_project/machinelearning_study/edgedriver_win64/msedgedriver.exe', options=options)
driver.get('')

