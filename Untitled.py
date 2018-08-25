# Testing engineer features Friday
%reload_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from LabelClass import LabelCountEncoder
from scipy.stats import skew
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from house import *
from config import *

def rmse_cv(model, x, y, k=5):
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_log_error", cv = k))
    return(np.mean(rmse))

def plot_results(prediction):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, prediction, s=20)
    plt.title('Predicted vs. Actual')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
    plt.tight_layout()

def rmsle(y_pred, y_test) :
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))


del house,house_sf
house = House('data/train.csv','data/test.csv')

# %% Clean data

house.cleanRP()

house.engineer_features(HOUSE_CONFIG)

# house.sg_ordinals()
# house.label_encode_engineer()

# house.all=pd.get_dummies(house.all,drop_first=True, dummy_na=True)

x=house.train().drop(['SalePrice','test'],axis=1)
y=house.train().SalePrice
x_train, x_test, y_train, y_test = train_test_split(x,y)

from sklearn.model_selection import GridSearchCV

alpha_100 =[{'alpha': np.logspace(1, 25, 50)}]
lasso = linear_model.Lasso(alpha=1,normalize=True)
# lasso.fit(x_train, y_train) # fit data
para_search = GridSearchCV(estimator=lasso,param_grid=alpha_100,scoring='neg_mean_squared_log_error',cv=5)
para_search.fit(x_train,y_train)
x_train

x_train.dtypes[x_train.dtypes==np.float64]

x_train.isnull().sum().sum()
y_train.isnull().sum()
rmsle(para_search.predict(x_test),y_test)

#R ^2 Coefficient
lasso.score(x_train,y_train)
rmse_cv(lasso,x_train,y_train)
rmsle(lasso.predict(x_test),y_test)
