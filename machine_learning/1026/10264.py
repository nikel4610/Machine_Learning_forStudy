import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

# 원본 행렬을 출력하고, SVD를 적용할 경우 U, Sigma, Vt 의 차원 확인
np.random.seed(121)
matrix = np.random.random((6, 6))
print('원본 행렬:\n',matrix)
# 원본 행렬:
#  [[0.11133083 0.21076757 0.23296249 0.15194456 0.83017814 0.40791941]
#  [0.5557906  0.74552394 0.24849976 0.9686594  0.95268418 0.48984885]
#  [0.01829731 0.85760612 0.40493829 0.62247394 0.29537149 0.92958852]
#  [0.4056155  0.56730065 0.24575605 0.22573721 0.03827786 0.58098021]
#  [0.82925331 0.77326256 0.94693849 0.73632338 0.67328275 0.74517176]
#  [0.51161442 0.46920965 0.6439515  0.82081228 0.14548493 0.01806415]]

U, Sigma, Vt = svd(matrix, full_matrices=False)
print('\n분해 행렬 차원:',U.shape, Sigma.shape, Vt.shape)
# 분해 행렬 차원: (6, 6) (6,) (6, 6)

print('\nSigma값 행렬:', Sigma)
# Sigma값 행렬: [3.2535007  0.88116505 0.83865238 0.55463089 0.35834824 0.0349925 ]

# Truncated SVD로 Sigma 행렬의 특이값을 4개로 하여 Truncated SVD 수행.
# 특이값 조절해서 차원 축소
num_components = 4
U_tr, Sigma_tr, Vt_tr = svds(matrix, k=num_components)
print('\nTruncated SVD 분해 행렬 차원:',U_tr.shape, Sigma_tr.shape, Vt_tr.shape)
# Truncated SVD 분해 행렬 차원: (6, 4) (4,) (4, 6)

print('\nTruncated SVD Sigma값 행렬:', Sigma_tr)
# Truncated SVD Sigma값 행렬: [0.55463089 0.83865238 0.88116505 3.2535007 ]
matrix_tr = np.dot(np.dot(U_tr,np.diag(Sigma_tr)), Vt_tr)  # output of TruncatedSVD

print('\nTruncated SVD로 분해 후 복원 행렬:\n', matrix_tr)
# Truncated SVD로 분해 후 복원 행렬:
#  [[0.19222941 0.21792946 0.15951023 0.14084013 0.81641405 0.42533093]
#  [0.44874275 0.72204422 0.34594106 0.99148577 0.96866325 0.4754868 ]
#  [0.12656662 0.88860729 0.30625735 0.59517439 0.28036734 0.93961948]
#  [0.23989012 0.51026588 0.39697353 0.27308905 0.05971563 0.57156395]
#  [0.83806144 0.78847467 0.93868685 0.72673231 0.6740867  0.73812389]
#  [0.59726589 0.47953891 0.56613544 0.80746028 0.13135039 0.03479656]]

