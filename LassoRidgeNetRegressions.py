"""
Author: Michael Kralis
Date: 08/28/2019
"""
#Import the necessary python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics

#Make sure the correct wd is selected to get the data
os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")

#Get the relevant data sets

MCD = pd.read_csv('MCD.csv',sep=',')
SP = pd.read_csv('Stock Performance.csv',sep=',')
SPY = pd.read_csv('SPY.csv',sep=',')

#Set up the next Regressions using the same independent data
X = np.column_stack((SP['Large BP'], SP['Large ROE'], SP['Large SP'], SP['Large Return Rate'], SP['Large Market Value'], SP['Small Systematic Risk']))
y = SP['Annual Return']
lm = LinearRegression()

#convert to numpy array and reshape
y = np.asarray(y)
y = np.reshape(y, (63,1))

#Start Multiregression
lm.fit(X,y)
y_fit = lm.predict(X)

#Print out test statistics
print("\nTest Statistics for Multi Regression against Annual Return")

print("\nThe coefficient for Large BP is:", lm.coef_[0][0])
print("The coefficient for Large ROE is:", lm.coef_[0][1])
print("The coefficient for Large SP is:", lm.coef_[0][2])
print("The coefficient for Large Return Rate is:", lm.coef_[0][3])
print("The coefficient for Large Market Value is:", lm.coef_[0][4])
print("The coefficient for Small Systematic Risk is:", lm.coef_[0][5])

print("\nR-Squared:" , metrics.explained_variance_score(y,y_fit))
print("MSE:", metrics.mean_squared_error(y,y_fit))
print("MAE:", metrics.mean_absolute_error(y,y_fit))

#Start new Ridge regression using same data
model = Ridge(alpha = 0.0005)
model.fit(X,y)
y_fit = model.predict(X)

#Print out test statistics
print("\nTest Statistics for Ridge Regression against Annual Return")

print("\nThe coefficient for Large BP is:", model.coef_[0][0])
print("The coefficient for Large ROE is:", model.coef_[0][1])
print("The coefficient for Large SP is:", model.coef_[0][2])
print("The coefficient for Large Return Rate is:", model.coef_[0][3])
print("The coefficient for Large Market Value is:", model.coef_[0][4])
print("The coefficient for Small Systematic Risk is:", model.coef_[0][5])

print("\nR-Squared:" , metrics.explained_variance_score(y,y_fit))
print("MSE:", metrics.mean_squared_error(y,y_fit))
print("MAE:", metrics.mean_absolute_error(y,y_fit))

#Start new Lasso Regression using same data
model = Lasso(alpha = 0.0005)
model.fit(X,y)
y_fit = model.predict(X)

#Print out test statistics
print("\nTest Statistics for Lasso Regression against Annual Return")

print("\nThe coefficient for Large BP is:", model.coef_[0])
print("The coefficient for Large ROE is:", model.coef_[1])
print("The coefficient for Large SP is:", model.coef_[2])
print("The coefficient for Large Return Rate is:", model.coef_[3])
print("The coefficient for Large Market Value is:", model.coef_[4])
print("The coefficient for Small Systematic Risk is:", model.coef_[5])

print("\nR-Squared:" , metrics.explained_variance_score(y,y_fit))
print("MSE:", metrics.mean_squared_error(y,y_fit))
print("MAE:", metrics.mean_absolute_error(y,y_fit))


#Set up regressions against Annual Return N
y = SP['Annual Return N']
y = np.asarray(y)
y = np.reshape(y, (63,1))

#Start Multiregression
lm.fit(X,y)
y_fit = lm.predict(X)

#Print out test statistics
print("\nTest Statistics for Multi Regression against Annual Return N")

print("\nThe coefficient for Large BP is:", lm.coef_[0][0])
print("The coefficient for Large ROE is:", lm.coef_[0][1])
print("The coefficient for Large SP is:", lm.coef_[0][2])
print("The coefficient for Large Return Rate is:", lm.coef_[0][3])
print("The coefficient for Large Market Value is:", lm.coef_[0][4])
print("The coefficient for Small Systematic Risk is:", lm.coef_[0][5])

print("\nR-Squared:" , metrics.explained_variance_score(y,y_fit))
print("MSE:", metrics.mean_squared_error(y,y_fit))
print("MAE:", metrics.mean_absolute_error(y,y_fit))

#Start new Ridge Regression using the same data
model = Ridge(alpha = 0.0005)
model.fit(X,y)
y_fit = model.predict(X)

#Print out test statistics
print("\nTest Statistics for Ridge Regression against Annual Return N")

print("\nThe coefficient for Large BP is:", model.coef_[0][0])
print("The coefficient for Large ROE is:", model.coef_[0][1])
print("The coefficient for Large SP is:", model.coef_[0][2])
print("The coefficient for Large Return Rate is:", model.coef_[0][3])
print("The coefficient for Large Market Value is:", model.coef_[0][4])
print("The coefficient for Small Systematic Risk is:", model.coef_[0][5])

print("\nR-Squared:" , metrics.explained_variance_score(y,y_fit))
print("MSE:", metrics.mean_squared_error(y,y_fit))
print("MAE:", metrics.mean_absolute_error(y,y_fit))

#Start new Lasso Regression using the same data
model = Lasso(alpha = 0.0005)
model.fit(X,y)
y_fit = model.predict(X)

#Print out test statistics
print("\nTest Statistics for Lasso Regression against Annual Return N")

print("\nThe coefficient for Large BP is:", model.coef_[0])
print("The coefficient for Large ROE is:", model.coef_[1])
print("The coefficient for Large SP is:", model.coef_[2])
print("The coefficient for Large Return Rate is:", model.coef_[3])
print("The coefficient for Large Market Value is:", model.coef_[4])
print("The coefficient for Small Systematic Risk is:", model.coef_[5])

print("\nR-Squared:" , metrics.explained_variance_score(y,y_fit))
print("MSE:", metrics.mean_squared_error(y,y_fit))
print("MAE:", metrics.mean_absolute_error(y,y_fit))


#Start MCD CAPM regression

#Create a lag column to calculate percent change in price
MCD['Lag'] = MCD['Adj Close'].shift(1)
SPY['Lag'] = SPY['Adj Close'].shift(1)

X = np.log(MCD['Lag']/MCD['Adj Close'])
y = np.log(SPY['Lag']/SPY['Adj Close'])

#Take out the 'NaN' piece of data
X.pop(0)
y.pop(0)

#Convert to numpy array and reshape
X = np.asarray(X)
y = np.asarray(y)
X = np.reshape(X,(251,1))
y = np.reshape(y, (251,1))

#Run the regression
lm = LinearRegression()
lm.fit(X,y)
a = lm.coef_[0]
b = lm.intercept_
y_fit = a*X + b

#Print out test statistics
print("\nThe stock CAPM beta for MCD is", a[0])
print("R-Squared:" , metrics.explained_variance_score(y,y_fit))
print("MSE:", metrics.mean_squared_error(y,y_fit))
print("MAE:", metrics.mean_absolute_error(y,y_fit))

#Plot the results
plt.title("MCD Regression")
plt.scatter(X,y, c='blue')
plt.plot(X,y_fit,c= 'red')