# Linear_regression25-pt2
Working with linear regression
``` python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# data = pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv') 
data = pd.read_csv('position_salaries.csv') 
data.head()

data.info()

#Observation: There are no null objects.
data.head()

X=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Observation: We have fit the linear regression model to the given dataset.

def viz_linear():
    plt.scatter(X,y, color = 'red')
    plt.plot(X,lin_reg.predict(X), color = 'blue')
    plt.title('Linear Regression Model')
    plt.xlabel('Position Level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_linear()

#Observation:In the above figure, you can see that even though we have fit the line, the data points are scattered and do not fit the line correctly.

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Higher degrees will increase the complexity of the model and increase overfitting of the data. 
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()

pol_reg.fit(X_poly, y)

#Observation: We have fitted the polynomial regression to the given data.

def viz_polynomial():
  plt.scatter(X, y, color = 'red')
  plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
  plt.title('Linear regression with polynomial with degree 4')
  plt.xlabel('Position level')
  plt.ylabel('Salary')
  plt.show()
  return

viz_polynomial()
#Observation: As you can see, the regression line is able to fit the majority of the data points.

lin_reg.predict([[5.5]])

#Observation: The linear regression model predicts the output would have been 249500.

pol_reg.predict(poly_reg.fit_transform([[5.5]]))

#Observations:From the above two outputs, it is clear that there is a difference between the two predictions.
#Hence, we can infer that non-linear inputs require non-linear models, such as the polynomial model.

```
Answers above, below is downloadable link empty file 
```
https://vocproxy-1-21.us-west-2.vocareum.com/files/home/labsuser/Block_25_Demo_2_Student.ipynb?_xsrf=2%7C3db90129%7Cacba5a54690658c9a4998717e4af4654%7C1708812522
```
