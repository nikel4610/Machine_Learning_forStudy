# 선형 회귀 실습

import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x 와 y의 평균값
mx = np.mean(x) # mx 는 x의 평균
my = np.mean(y) # my 는 y의 평균
print("x와 y의 평균값: ",mx , my)

# 기울기 공식의 분모
divisor = sum([(mx - i) ** 2 for i in x])

#기울기 공식의 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my) # x 와 y 의 편차를 곱해서 합
    return d
dividend = top(x, mx, y, my)

print("분모, 분자: ", divisor, dividend)

# 기울기 (a) 와 y절편 (b) 구하기
a = dividend / divisor
b = my - (mx * a)
print("기울기 a, y 절편 b: ", a, b)

# x와 y의 평균값:  5.0 90.5
# 분모, 분자:  20.0 46.0
# 기울기 a, y 절편 b:  2.3 79.0
