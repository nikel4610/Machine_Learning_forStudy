from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 임의의 데이터 생성
X = np.arange(4).reshape(2, 2)
print(X)
# [[0 1]
#  [2 3]]

# degree = 2인 2차 다항식으로 변환
poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_f = poly.transform(X)
print(poly_f)
# [[1. 0. 1. 0. 0. 1.]
#  [1. 2. 3. 4. 6. 9.]]

def polynomial_func(X):
    y = 1 + 2*X + X**3
    return y
print(polynomial_func(X))
# [[ 1  4]
#  [13 34]]

# 3차 다항식 변환
poly_f = PolynomialFeatures(degree=3).fit_transform(X)
print(poly_f)
# [[ 1.  0.  1.  0.  0.  1.  0.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.  8. 12. 18. 27.]]

# Linear Regression에 3차 다항식 계수 feature와 3차 다항식 결정값으로 학습 후 회귀 게수 확인
model = LinearRegression()
y = polynomial_func(X)
model.fit(poly_f, y)

print('Polynomial 회귀 계수: ', np.round(model.coef_, 2))
# Polynomial 회귀 계수:  [[0.   0.02 0.02 0.04 0.05 0.07 0.07 0.11 0.16 0.23]
#                        [0.   0.05 0.05 0.09 0.14 0.18 0.18 0.27 0.41 0.59]]
print('Polynomial 회귀 shape: ', model.coef_.shape)
# Polynomial 회귀 shape:  (2, 10)

# 파이프라인을 이용한 3차 다항 회귀 실습
from sklearn.pipeline import Pipeline

def poly_func(X):
    y = 1 + 2*X + X**2 + X**3
    return y

model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())])

X = np.arange(4).reshape(2, 2)
y = poly_func(X)

model = model.fit(X, y)
print('Polynomial 회귀 계수: ', np.round(model.named_steps['linear'].coef_, 2))
# Polynomial 회귀 계수:  [[0.   0.02 0.02 0.05 0.07 0.1  0.1  0.14 0.22 0.31]
#                        [0.   0.06 0.06 0.11 0.17 0.23 0.23 0.34 0.51 0.74]]