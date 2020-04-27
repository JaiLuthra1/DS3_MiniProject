'''
PCA applied, now Regression (Linear)
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
def train_test_split(data,test_size,seed): return model_selection.train_test_split(data.iloc[:,0:-1],data.iloc[:,-1:],test_size=test_size,random_state=seed,shuffle=True)
def rms(y_test,y_pred): return sqrt(mean_squared_error(y_test,y_pred))
a=["CSV_Files/dim_reduced_to_1.csv","CSV_Files/dim_reduced_to_2.csv","CSV_Files/dim_reduced_to_3.csv","CSV_Files/dim_reduced_to_4.csv","CSV_Files/dim_reduced_to_5.csv","CSV_Files/dim_reduced_to_6.csv","CSV_Files/standardised.csv","CSV_Files/normalized.csv"]
for i in range(len(a)):
    data=pd.read_csv(a[i])
    print("For ",a[i])
    data.drop("CreationTime",inplace=True,axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(data,0.3,42)
    regressor=LinearRegression()
    regressor.fit(X_train,Y_train)
    Y_pred_test=regressor.predict(X_test)
    plt.scatter(Y_test,Y_pred_test)
    plt.xlabel("Y_test")
    plt.ylabel("Y_pred_test")
    plt.show()
    print("R2 score for test data :",r2_score(Y_test.values,Y_pred_test))
    print("rmse score for test data :",rms(Y_test.values,Y_pred_test))
    Y_pred_train=regressor.predict(X_train)
    plt.scatter(Y_train,Y_pred_train)
    plt.xlabel("Y_train")
    plt.ylabel("Y_pred_train")
    plt.show()
    print("R2 score for train data :",r2_score(Y_train.values,Y_pred_train))
    print("rmse score for train data :",rms(Y_train.values,Y_pred_train))
    print("Model coefficients :")
    print(regressor.coef_)