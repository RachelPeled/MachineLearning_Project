
## Data loading
# %%
%reload_ext autoreload
%autoreload 2

from house import *
from config import *

del house
house = House('data/train.csv','data/test.csv')
#house_rp=House('data/train.csv','data/test.csv')
# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
import matplotlib.pyplot as plt
from scipy.special import boxcox1p


house.cleanRP()
house.all['AllSF'] = house.all['GrLivArea'] + house.all['TotalBsmtSF'] + house.all['1stFlrSF'] + house.all['2ndFlrSF']
plt.figure(figsize=(10,6))
plt.scatter(house.all['AllSF'],house.all['SalePrice'])
plt.xlabel('Total SF')
plt.ylabel('Sale Price')
z = np.polyfit(house.all['AllSF'],house.all['SalePrice'], 1)
p = np.poly1d(z)
plt.plot(house.all['AllSF'],p(house.all['AllSF']),"r--")
plt.show()


house.all=house.all.drop([1298,523])

house.all['AllSF'].corr(house.all['SalePrice'])
house.all[house.all['AllSF']>12000]







house.convert_types(HOUSE_CONFIG)



skewness = house.all.select_dtypes(exclude = ["object"]).apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
skewed_features

for i in range(0,len(skewed_features)):
    house.all[skewed_features[i]]=house.all[skewed_features[i]]+1
    house.all[skewed_features[i]]=house.all[skewed_features[i]].apply(np.log)



house.engineer_features(HOUSE_CONFIG)
##################################################

train=house.dummy_train[house.dummy_train['test']==False].drop('test',axis=1)
test=house.dummy_train[house.dummy_train['test']==True].drop(['test','SalePrice'],axis=1)
y_train=train['SalePrice']
x_train=train.drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split( x_train, y_train)





############### Lasso model
lasso = linear_model.Lasso(normalize=True) # create a lasso instance
lasso.fit(X_train, Y_train)# fit data
print("The determination of Lasso regression is: %.4f" %lasso.score(X_train, Y_train))

y_pred=lasso.predict(X_test)
np.sqrt(np.mean((np.log1p(y_pred) - np.log1p( Y_test))**2))


====================================


######### feature Engeneneering

#0.9181

#with feature 0.9279
#removing outliers 0.9288
#0.1226
======================================





train=house.dummy_train[house.dummy_train['test']==False].drop('test',axis=1)
test=house.dummy_train[house.dummy_train['test']==True].drop(['test','SalePrice'],axis=1)
y_train=train['SalePrice']
x_train=train.drop('SalePrice',axis=1)
X_train, X_test, Y_train, Y_test = train_test_split( x_train, y_train)
Y_train=Y_train.apply(np.log)
lasso = linear_model.Lasso(normalize=True)
grid_param = [{'alpha': np.logspace(-4, 2, 100)}]
para_search_lasso = GridSearchCV(estimator=lasso, param_grid=grid_param, scoring='neg_mean_squared_error', cv=5, return_train_score=True)
para_search_lasso.fit(X_train, Y_train)
y_pred=para_search_lasso.predict(X_test)
np.exp(y_pred)
print(para_search_lasso.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(para_search_lasso.best_score_)))
np.sqrt(np.mean((np.log1p(np.exp(y_pred)) - np.log1p( Y_test))**2))
test

ID=pd.DataFrame(house.Id)['Id']
predict=pd.DataFrame(para_search_lasso.predict(test),columns=['SalePrice'])
predict=np.exp(predict)
predict
predict['ID']=list(range(1461,2920))
predict=predict[['ID','SalePrice']]
predict.to_csv('predict3',index=False)
