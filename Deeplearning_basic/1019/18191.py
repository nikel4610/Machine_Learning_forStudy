import nltk
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from konlpy.corpus import kobill

from matplotlib import font_manager , rc
from tensorflow.keras.preprocessing.text import text_to_word_sequence

path = "c:/Windows/Fonts/malgun.ttf"
font_name= font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)

doc = kobill.open('1809890.txt').read()
t = Okt()
tokens = t.nouns(doc)
stop_words = ['.', '(', ')', ',', '의', '자', '에', '안', '번', '호', '을', '이', '다', '및', '명', '것', '중', '안', '위', '만', '액', '제', '표', '수', '월', '세', '생', '략', '함', '정']

ko = nltk.Text(tokens, name='대한민국 국회 의안 제 1809890호')
ko = [each_word for each_word in ko if each_word not in stop_words]
ko = nltk.Text(ko, name='대한민국 국회 의안 제 1809890호')

plt.figure(figsize=(10,8))
ko.plot(50)
plt.show()