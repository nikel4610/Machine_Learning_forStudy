# 보스턴 집값 예측
# 릿지, 라쏘, 엘라스틱넷을 사용하여 보스턴 집값 예측
# TODO -> 데이터 다시 확인하기 *보스턴 csv 파일이 안됨

# # 각 alpha에 따른 회귀 계수 값을 시각화하기 위해 5개의 열로 된 맷플롯립 축 생성
# fig , axs = plt.subplots(figsize=(18,6) , nrows=1 , ncols=5)
# # 각 alpha에 따른 회귀 계수 값을 데이터로 저장하기 위한 DataFrame 생성
# coeff_df = pd.DataFrame()
#
# # alphas 리스트 값을 차례로 입력해 회귀 계수 값 시각화 및 데이터 저장. pos는 axis의 위치 지정
# for pos , alpha in enumerate(alphas) :
#     ridge = Ridge(alpha = alpha)
#     ridge.fit(X_data , y_target)
#     # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가.
#     coeff = pd.Series(data=ridge.coef_ , index=X_data.columns )
#     colname='alpha:'+str(alpha)
#     coeff_df[colname] = coeff
#     # 막대 그래프로 각 alpha 값에서의 회귀 계수를 시각화. 회귀 계수값이 높은 순으로 표현
#     coeff = coeff.sort_values(ascending=False)
#     axs[pos].set_title(colname)
#     axs[pos].set_xlim(-3,6)
#     sns.barplot(x=coeff.values , y=coeff.index, ax=axs[pos])
#
# # for 문 바깥에서 맷플롯립의 show 호출 및 alpha에 따른 피처별 회귀 계수를 DataFrame으로 표시
# plt.show()

# ------------------------------------------------------------------------------------------

# from sklearn.linear_model import Lasso, ElasticNet
#
# # alpha값에 따른 회귀 모델의 폴드 평균 RMSE를 출력하고 회귀 계수값들을 DataFrame으로 반환
# def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None,
#                         verbose=True, return_coeff=True):
#     coeff_df = pd.DataFrame()
#     if verbose : print('####### ', model_name , '#######')
#     for param in params:
#         if model_name =='Ridge': model = Ridge(alpha=param)
#         elif model_name =='Lasso': model = Lasso(alpha=param)
#         elif model_name =='ElasticNet': model = ElasticNet(alpha=param, l1_ratio=0.7)
#         neg_mse_scores = cross_val_score(model, X_data_n,
#                                              y_target_n, scoring="neg_mean_squared_error", cv = 5)
#         avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
#         print('alpha {0}일 때 5 폴드 세트의 평균 RMSE: {1:.3f} '.format(param, avg_rmse))
#         # cross_val_score는 evaluation metric만 반환하므로 모델을 다시 학습하여 회귀 계수 추출
#
#         model.fit(X_data_n , y_target_n)
#         if return_coeff:
#             # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가.
#             coeff = pd.Series(data=model.coef_ , index=X_data_n.columns )
#             colname='alpha:'+str(param)
#             coeff_df[colname] = coeff
#
#     return coeff_df
# # end of get_linear_regre_eval