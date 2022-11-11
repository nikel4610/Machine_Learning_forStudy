from sklearn.datasets import load_iris

iris_data = load_iris()

# print(type(iris_data))
# <class 'sklearn.utils._bunch.Bunch'>

keys = iris_data.keys()

# print(keys)
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

print('feature_name type: ', type(iris_data.feature_names))
print('feature_name shape: ', len(iris_data.feature_names))
print(iris_data.feature_names)

# feature_name type:  <class 'list'>
# feature_name shape:  4
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print('target_name type: ', type(iris_data.target_names))
print('target_name shape: ', len(iris_data.target_names))
print(iris_data.target_names)

# target_name type:  <class 'numpy.ndarray'>
# target_name shape:  3
# ['setosa' 'versicolor' 'virginica']

print('data type: ', type(iris_data.data))
print('data shape: ', iris_data.data.shape)
print(iris_data['data'])

# data type:  <class 'numpy.ndarray'>
# data shape:  (150, 4)
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  ...
#  [6.2 3.4 5.4 2.3]
#  [5.9 3.  5.1 1.8]]

print('target type: ', type(iris_data.target))
print('target shape: ', iris_data.target.shape)
print(iris_data.target)

# target type:  <class 'numpy.ndarray'>
# target shape:  (150,)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]