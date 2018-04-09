import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
x[:,3] = label_encoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x= onehotencoder.fit_transform(x).toarray()

#split train test data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state =0)

#fit simple linear reg to training model
from sklearn.linear_model import LinearRegression 
regressor =  LinearRegression()
regressor.fit(x_train,y_train)

#predict test set 
y_pred = regressor.predict(x_test)

#using optimal model backward elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int),values = x, axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:,[0,3,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:,[0,3,5]]
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()