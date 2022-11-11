# SVD 중요 (자료 361p)
import numpy as np
from numpy.linalg import svd

# 4x4 랜덤 행렬 생성
np.random.seed(121)
a = np.random.randn(4,4)

U, sigma, Vt = svd(a)
print(U.shape, sigma.shape, Vt.shape)
# (4, 4) (4,) (4, 4)

print('U matrix:\n', np.round(U, 3))
# U matrix:
#  [[-0.079 -0.318  0.867  0.376]
#  [ 0.383  0.787  0.12   0.469]
#  [ 0.656  0.022  0.357 -0.664]
#  [ 0.645 -0.529 -0.328  0.444]]

print('Sigma Value:\n', np.round(sigma, 3))
# Sigma Value:
#  [3.423 2.023 0.463 0.079]

print('V transpose matrix:\n', np.round(Vt, 3))
# V transpose matrix:
#  [[ 0.041  0.224  0.786 -0.574]
#  [-0.2    0.562  0.37   0.712]
#  [-0.778  0.395 -0.333 -0.357]
#  [-0.593 -0.692  0.366  0.189]]

# sigma를 다시 0을 포함한 대칭행렬로 변환
sigma_mat = np.diag(sigma)
a_ = np.dot(np.dot(U, sigma_mat), Vt)
print('A matrix:\n', np.round(a_, 3))
# A matrix:
#  [[-0.212 -0.285 -0.574 -0.44 ]
#  [-0.33   1.184  1.615  0.367]
#  [-0.014  0.63   1.71  -1.327]
#  [ 0.402 -0.191  1.404 -1.969]]

a[2] = a[0] + a[1]
a[3] = a[0]
print(np.round(a, 3))
# [[-0.212 -0.285 -0.574 -0.44 ]
#  [-0.33   1.184  1.615  0.367]
#  [-0.542  0.899  1.041 -0.073]
#  [-0.212 -0.285 -0.574 -0.44 ]]

# 다시 SVD를 수행하여 sigma 값 확인
U, sigma, Vt = svd(a)
print(U.shape, sigma.shape, Vt.shape)
# (4, 4) (4,) (4, 4)

print('Sigma Value:\n', np.round(sigma, 3))
# Sigma Value:
#  [2.663 0.807 0.    0.   ]

# U 행렬의 경우 sigma와 내적을 수행하므로 sigma의 앞 2행에 대응되는 앞 2열만 추출
U_ = U[:, :2]
sigma_ = np.diag(sigma[:2])

# V 전치 행렬의 경우 앞 2행만 추출
Vt_ = Vt[:2, :]
print(U_.shape, sigma_.shape, Vt_.shape)
# (4, 2) (2, 2) (2, 4)

# U, sigma, Vt의 내적을 수행하며, 다시 원본 행렬 복원
a_ = np.dot(np.dot(U_, sigma_), Vt_)
print(np.round(a_, 3))
# [[-0.212 -0.285 -0.574 -0.44 ]
#  [-0.33   1.184  1.615  0.367]
#  [-0.542  0.899  1.041 -0.073]
#  [-0.212 -0.285 -0.574 -0.44 ]]

