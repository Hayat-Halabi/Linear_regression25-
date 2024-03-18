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
# Ridge Regression 
Houseing - data
``` python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

df= pd.read_csv('housing.csv')
df.head()

#Observations: This is the head of the data. The column MEDV is the target variable.

df.info()

X = df.drop(['MEDV'], axis = 1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
Ridge_model = Ridge(alpha=1).fit(X_train, y_train)
Ridge_model.intercept_

#Observation: The intercept value for the Ridge model is 24.8.

y_pred = Ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Observation: The RMSE value is 4.74.

Ridge_model.coef_
r2_score(y_test, y_pred)

#Observations: The given model is a moderate fit for the given data.

from sklearn.model_selection import GridSearchCV
# RepeatedKFold is a cross-validation technique 
cv = RepeatedKFold(n_splits =10, n_repeats =3, random_state =1)
grid = dict()
grid['alpha'] = np.arange(0,1,0.1)
model = Ridge()
search = GridSearchCV(model, grid, scoring = 'neg_mean_absolute_error',cv = cv, n_jobs= -1)

results = search.fit(X_train, y_train)
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

# Observation: As you can see from the output, the score is -3.5, and the configuration fusion alpha is 0.7.

Ridge_model = Ridge (alpha = 0.7).fit(X_train, y_train)
y_pred = Ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

r2_score(y_test,y_pred)

#Observation:The r2 score is 0.68, which means it is a moderate fit for the given data.

pd.Series(Ridge_model.coef_, index = X_train.columns)

#Observation:A lot of penalization has occurred, as you can see in the negative values as well.

```
# Lasso Regression
Hitters CSV Lasso 
``` python


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV 
import matplotlib.pyplot as plt


df = pd.read_csv('Hitters.csv')
df.head()
df.info()
#Observations Overall, it includes 322 observations and 22 columns. Notice that there are some null values in Salary.

df['Salary'].fillna(df['Salary'].median(skipna=True), inplace=True)
df.isna().sum() # Confirm whether there are any null values left in the dataset.

#Observation As shown, it is quite clear from the output that the missing values have been replaced.

dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']], drop_first=True)
y = df['Salary']
x_ = df.drop(['Unnamed: 0', 'Salary', 'League', 'Division', 'NewLeague'], axis =1).astype('float64')
X = pd.concat([x_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis =1)


X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


lasso_model = Lasso(alpha=0.1, max_iter=10000).fit(X_train, y_train)
lasso_model.intercept_

#Observation: The intercept value is 344.

lasso_model.coef_

#Observation: In lasso regression, the attribute will be used, and it is evident that the last attribute, NewLeague, has been penalized to zero.

y_pred = lasso_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#Observation: RMSE is 345.53

r2_score(y_test, y_pred)
#Observation: r2 score is 0.36.

#Now, try to optimize using cross-validation.
lasso_cv_model = LassoCV(alphas = np.random.randint(0, 1000, 100), cv =10, max_iter=10000, n_jobs=-1).fit(X_train, y_train)

lasso_cv_model.alpha_

#Observation: This is the best alpha value that we got from cross-validation.


lasso_tuned = Lasso(alpha=7, max_iter=10000).fit(X_train, y_train) # Insert the best alpha value in the alpha parameter.
y_pred_tuned = lasso_tuned.predict(x_test) 
np.sqrt(mean_squared_error(y_test, y_pred_tuned))

#Observations : As you can see above, RMSEs are more, It has increased by 1%.

pd.Series(lasso_tuned.coef_, index=X_train.columns)

#Observations: As compared with the normal lasso, you can see that it has penalized the other group attributes like League and RBI.
# This is the penalization process of the lasso regression algorithm.
```
Answers above, below is downloadable link empty file 
```
https://vocproxy-1-21.us-west-2.vocareum.com/files/home/labsuser/Block_25_Demo_2_Student.ipynb?_xsrf=2%7C3db90129%7Cacba5a54690658c9a4998717e4af4654%7C1708812522
```
Link to access housing csv
```
https://vocproxy-1-21.us-west-2.vocareum.com/files/home/labsuser/housing.csv?_xsrf=2%7C3db90129%7Cacba5a54690658c9a4998717e4af4654%7C1708812522
```
Link to access Hitters_CSV
```
https://vocproxy-1-21.us-west-2.vocareum.com/files/home/labsuser/Hitters.csv?_xsrf=2%7C3db90129%7Cacba5a54690658c9a4998717e4af4654%7C1708812522
```
