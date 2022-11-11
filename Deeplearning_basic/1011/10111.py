import matplotlib.pyplot as plt
import numpy as np

# # 변수 a, p, q값을 임의로 정하고 x에 대한 y그래프 그리기
# # x가 1일때 y의 최소값이 3이 되는 식
# a = 1
# p = 1
# q = 3

# x = np.linspace(-4, 8, 100)
# y = a*(x-p)**2 + q

# plt.figure(figsize=(8,8))     
# plt.plot(x, y)

# plt.axhline(0,color='black')    
# plt.axvline(0,color='black')   
# plt.grid()
# plt.show()  

# # 공부한 시간과 점수를 각각 x, y라는 이름의 넘파이 배열로 만듭니다.
# # X = np.array([2, 4, 6, 8])
# # y1 = np.array([81, 93, 91, 97]) 
# # y2 = np.array([74, 80, 82, 87])
# X = np.array([2, 3, 4, 6, 8])
# y1 = np.array([81, 83, 93, 91, 97])

# y = 2.3*X + 79

# # 공부한 시간(x) vs. 취득 성적(y)의 그래프를 그립니다.
# plt.plot(X,y1,'o')
# # plt.plot(X,y2,'o')
# plt.plot(X,y)
# plt.xlabel('study time')
# plt.grid()
# plt.show()

x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 
hat_y = np.array([84.6, 89.2, 93.8, 98.4])

mse = 1/len(x) * sum( (y-hat_y)**2 )
print('평균제곱오차(MSE):', mse)

