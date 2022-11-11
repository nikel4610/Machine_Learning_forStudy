import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# split the data into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

# Create linear regression object
model = LinearRegression()
model.fit(x_train, y_train)

# prediction
y_pred = model.predict(x_test)

plt.plot(y_test, y_pred, 'o')
x = np.linspace(0, 350, 100)
y = x
plt.plot(x, y)
plt.show()