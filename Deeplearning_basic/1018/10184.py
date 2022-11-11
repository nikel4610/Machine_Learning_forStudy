import os
import sys
import requests
import json
import urllib.request
import matplotlib.pyplot as plt

client_id = json.loads(open('D:/vsc_project/machinelearning_study/api.json').read())['client_id']
client_secret = json.loads(open('D:/vsc_project/machinelearning_study/api.json').read())['client_secret']
url = "https://openapi.naver.com/v1/datalab/search"
body = "{\"startDate\":\"2021-01-01\",\"endDate\":\"2021-12-31\",\"timeUnit\":\"month\",\"keywordGroups\":[{\"groupName\":\"ㅇ\",\"keywords\":[\"난로\"]}],\"device\":\"pc\",\"ages\":[\"1\",\"2\"],\"gender\":\"f\"}";

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
request.add_header("Content-Type","application/json")

response = urllib.request.urlopen(request, data=body.encode("utf-8"))
rescode = response.getcode()

if(rescode==200):
    response_body = response.read()
else:
    print("Error Code:" + rescode)

type(eval(response_body.decode('utf-8')))
rData = eval(response_body.decode('utf-8'))
print(rData)

x = []
y = []

for i in rData['results'][0]['data']:
    x.append(i['period'])
    y.append(i['ratio'])

plt.plot(x, y)
plt.title('KeyWord Search Ratio per Month')
plt.xlabel('Month')
plt.ylabel('Ratio')
plt.show()