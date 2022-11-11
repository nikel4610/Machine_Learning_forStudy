import bs4
from bs4 import BeautifulSoup
import urllib.request as req

# url = 'https://glee-glee.com/'
url = 'https://finance.naver.com/marketindex/'
html = req.urlopen(url)
soup = BeautifulSoup(html, 'html.parser')
#
# # 상품명 추출
# title = soup.find_all('span', {'class':'name'})
# for i in title:
#     print(i.text)

# # CSS 선택자 사용
# html = """
# <html><body>
#   <div id = “m123”>
#     <h1>도서</h1>
#     <ul class = “items”>
#       <li>책 1</li>
#       <li>책 2</li>
#       <li>책 3</li>
#     </ul>
#   </div>
#   <div id = “m124”>
#     <h1> 음반 </h1>
#     <ul class = “music”>
#       <li> 나훈아 – 테스형</li>
#       <li> 구창모 – 희나리</li>
#       <li> 에이프릴 – 예쁜 게 죄</li>
#     </ul>
#   </div>
# </body></html>
# """
#
# soup = BeautifulSoup(html, "html.parser")
# h1 = soup.select_one("div#meign > h1").string
# # ‘#’은 id 선택
#
# lst = soup.select("div#megin > ul.items > li")
#
# for li in lst:
#   print("li = ", li.string)

# #exchangeList > li.on > a.head.usd > div > span.value
price = soup.select_one('div.head_info > span.value').string
print("미국환율: ", price)

