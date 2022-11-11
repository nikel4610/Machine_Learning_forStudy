from bs4 import BeautifulSoup
import urllib.request as req

# html = """
# <html><body>
#   <h1 id="title1">스크레이핑이란?</h1>
#   <p id = "body1">웹 페이지를 분석하는 것</p>
#   <p>원하는 부분을 추출하는 것</p>
#
#   <h1 id="title2">스크레이핑이란?</h1>
#   <p id = "body2">웹 페이지를 분석하는 것</p>
#   <p>원하는 부분을 추출하는 것</p>
# </body></html>
# """
#
# soup = BeautifulSoup(html, 'html.parser')
# title = soup.find(id="title2")
# body = soup.find(id="body2")
#
# print(title)
# # <h1 id="title2">스크레이핑이란?</h1>
# print(title.string)
# # 스크레이핑이란?
# print(body)
# # <p id="body2">웹 페이지를 분석하는 것</p>
# print(body.string)
# # 웹 페이지를 분석하는 것

# html = """
# <html>
#     <body>
#         <ul class="greet">
#             <li>hello</li>
#             <li>bye</li>
#             <li>welcome</li>
#         </ul>
#         <ul class="reply">
#             <li>ok</li>
#             <li>no</li>
#             <li>sure</li>
#         </ul>
# </body>
# </html>
# """
#
# soup = BeautifulSoup(html, "html.parser")
#
# ul_class_reply = soup.find('ul', {'class':'reply'}) # ul 태그 중 class가 reply인 것
# li_tags = ul_class_reply.findAll('li') # ul 태그 중 class가 reply인 것의 li 태그
#
# for li in li_tags:
#     print(li)
#     # <li>ok</li>
#     # <li>no</li>
#     # <li>sure</li>
#
# for li in li_tags:
#     print(li.text)
#     # ok
#     # no
#     # sure

url = 'https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=cur&date=20221016'
html = req.urlopen(url)
soup = BeautifulSoup(html, 'html.parser')

# ul_tag = soup.find('ul', {'class':'list_nav NM_FAVORITE_LIST'})
# li_tags = ul_tag.findAll('li')
#
# for li_tag in li_tags:
#     print(li_tag.string)

# titleOfMovie = soup.find_all('div', {'class':'tit5'})
# points = soup.find_all('td', {'class':'point'})
#
# title = []
# for i in titleOfMovie:
#     title.append(i.find('a').text)
#
# point = []
# for i in points:
#     point.append(i.text)
#
# for i in range(0, len(title)):
#     print(str(i+1) + ':' + title[i] + '[' + point[i] + ']')
#     # 1:탑건: 매버릭[9.77] ...

# 2022년 10월 1일부터 16일까지의 날짜별 영화제목 및 평점 출력
for i in range(1, 17):
    url = 'https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=cur&date=202210' + str(i).zfill(2)
    html = req.urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    titleOfMovie = soup.find_all('div', {'class':'tit5'})
    points = soup.find_all('td', {'class':'point'})

    title = []
    for i in titleOfMovie:
        title.append(i.find('a').text)

    point = []
    for i in points:
        point.append(i.text)

    for i in range(0, len(title)):
        print(str(i+1) + ':' + title[i] + '[' + point[i] + ']')
    print('----------------------------------')