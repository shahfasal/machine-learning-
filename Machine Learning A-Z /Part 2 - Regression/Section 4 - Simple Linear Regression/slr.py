# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#split train test data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=1/3, random_state =0)

'''#missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ='NaN', strategy = 'mean', axis = 0 )
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])'''


#fit simple linear reg to training model
from sklearn.linear_model import LinearRegression 
regressor =  LinearRegression()
regressor.fit(x_train,y_train)


#predict test set 
y_pred = regressor.predict(x_test)

#visualise training set results
plt.scatter(x_train,y_train,color ='red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('salary vs experience(training set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

#visualise trest set results
plt.scatter(x_test,y_test,color ='red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('salary vs experience(test set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

