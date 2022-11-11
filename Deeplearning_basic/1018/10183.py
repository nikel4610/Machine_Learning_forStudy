import os
import sys
import requests
import json

client_id = json.loads(open('D:/vsc_project/machinelearning_study/api.json').read())['client_id']
client_secret = json.loads(open('D:/vsc_project/machinelearning_study/api.json').read())['client_secret']
url =  "https://openapi.naver.com/v1/search/news.json"

headers = {
    'X-Naver-Client-Id': client_id,
    'X-Naver-Client-Secret': client_secret
}

payload = {
    'query': '안동역',
    'display': 40,
    'start': 1,
    'sort': 'date'
}

response = requests.get(url, headers=headers, params=payload)
dic_res = response.json()

for item in dic_res['items']:
    print( dic_res['items'].index(item)+1, item['title'], item['link'] )
