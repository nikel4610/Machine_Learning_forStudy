import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('D:/vsc_project/machinelearning_study/pizza.csv')

df['price_rupiah'] = df['price_rupiah'].str.replace('Rp', '').str.replace(',', '').astype('float64') # Remove Rp
df['diameter'] = df['diameter'].str.replace('inch', '').str.replace(',', '').astype('float64') # Remove Inch
df.dropna(inplace=True)

# log 값 변환 시 NaN등의 이슈로 log() 가 아닌 log1p() 를 이용하여 RMSLE 계산
def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle
# 사이킷런의 mean_square_error() 를 이용하여 RMSE 계산
def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred))
# MSE, RMSE, RMSLE 를 모두 계산
def evaluate_regr(y,pred):
    rmsle_val = rmsle(y,pred)
    rmse_val = rmse(y,pred)
    # MAE 는 scikit learn의 mean_absolute_error() 로 계산
    mae_val = mean_absolute_error(y,pred)
    print('RMSLE: {0:.3f}, RMSE: {1:.3F}, MAE: {2:.3F}'.format(rmsle_val, rmse_val, mae_val))

# labelencoding
encoder = LabelEncoder()

for i in df.columns:
    if df[i].dtype == 'object':
        df[i] = encoder.fit_transform(df[i])

    if df[i].dtype == 'float64':
        df[i] = df[i].astype('int64')

# print(df.head())
# print(df.info())

y_target = df['price_rupiah']
x_data = df.drop(['price_rupiah'], axis=1, inplace=False)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_target, test_size=0.3, random_state=42)
# print(df.info())

lr_reg = LinearRegression()
lr_reg.fit(x_train, y_train)
pred = lr_reg.predict(x_test)

evaluate_regr(y_test ,pred)
# RMSLE: 0.362, RMSE: 26285.581, MAE: 18311.492

# log변형
y_target_log = np.log1p(y_target)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_target_log, test_size=0.3, random_state=42)

lr_reg = LinearRegression()
lr_reg.fit(x_train, y_train)
pred = lr_reg.predict(x_test)

y_test_exp = np.expm1(y_test)
pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp ,pred_exp)
# RMSLE: 0.319, RMSE: 26478.005, MAE: 19208.318

coef = pd.Series(lr_reg.coef_, index=x_data.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort.values, y=coef_sort.index)
plt.show()

# One Hot Encoding
x_features_ohe = pd.get_dummies(x_data, columns=['topping', 'size', 'variant', 'diameter', 'extra_sauce', 'extra_cheese', 'extra_mushrooms'])
# 원-핫 인코딩이 적용된 feature 데이터 세트 기반으로 학습/예측 데이터 분할.
X_train, X_test, y_train, y_test = train_test_split(x_features_ohe, y_target_log,
                                                    test_size=0.3, random_state=0)

# 모델과 학습/테스트 데이터 셋을 입력하면 성능 평가 수치를 반환
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1 :
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print('###',model.__class__.__name__,'###')
    evaluate_regr(y_test, pred)

# model 별로 평가 수행
lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=10)
lasso_reg = Lasso(alpha=0.01)

for model in [lr_reg, ridge_reg, lasso_reg]:
    get_model_predict(model,X_train, X_test, y_train, y_test,is_expm1=True)
# ### LinearRegression ###
# RMSLE: 0.212, RMSE: 16862.361, MAE: 10908.071
# ### Ridge ###
# RMSLE: 0.245, RMSE: 15419.711, MAE: 13272.548
# ### Lasso ###
# RMSLE: 0.222, RMSE: 14139.599, MAE: 10819.388

coef = pd.Series(lr_reg.coef_ , index=x_features_ohe.columns)
coef_sort = coef.sort_values(ascending=False)[:10]
sns.barplot(x=coef_sort.values , y=coef_sort.index)
plt.show()

# 랜덤 포레스트, GBM, XGBoost, LightGBM model 별로 평가 수행
rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)

for model in [rf_reg, gbm_reg]:
    # XGBoost의 경우 DataFrame이 입력 될 경우 버전에 따라 오류 발생 가능. ndarray로 변환.
    get_model_predict(model,X_train.values, X_test.values, y_train.values, y_test.values,is_expm1=True)
# ### RandomForestRegressor ###
# RMSLE: 0.190, RMSE: 14279.476, MAE: 9427.808
# ### GradientBoostingRegressor ###
# RMSLE: 0.144, RMSE: 9139.397, MAE: 6105.279

# 예측 가격과 실제 가격 비교 그래프
test = pd.DataFrame({'predict_price':pred_exp, 'actual_price':y_test_exp})
test = test.reset_index()
test = test.drop(['index'], axis=1)
plt.figure(figsize=(10,10))
sns.lineplot(data=test)
plt.legend()
plt.show()