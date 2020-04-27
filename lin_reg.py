'''
Outliers Removed, now Regression (Linear)
'''

import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
def rms(y_test,y_pred): return sqrt(mean_squared_error(y_test,y_pred))

data=pd.read_csv("CSV_Files/outliers_removed.csv")
col=data["InBandwidth"]
data=data[["OutBandwidth","InTotalPPS","OutTotalPPS"]]
X_train, X_test, Y_train, Y_test = train_test_split(data,col,test_size=0.3, random_state=42)
lin_reg=LinearRegression()
lin_reg.fit(X_train,Y_train)
Y_pred_test=lin_reg.predict(X_test)
plt.scatter(Y_test,Y_pred_test,edgecolors='k',linewidths=0.5)
plt.xlabel("Y_test")
plt.ylabel("Y_pred_test")
print("RMSE is {}".format(rms(Y_test,Y_pred_test)))
print("R2 score is: ",r2_score(Y_test.values,Y_pred_test))
plt.show()
Y_pred_train=lin_reg.predict(X_train)
plt.scatter(Y_train,Y_pred_train,edgecolors='k',linewidths=0.5)
plt.xlabel("Y_train")
plt.ylabel("Y_pred_train")
print("RMSE is {}".format(rms(Y_train,Y_pred_train)))
print("R2 score is: ",r2_score(Y_train.values,Y_pred_train))
plt.show()
