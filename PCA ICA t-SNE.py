"""
@author: Michael Kralis
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import model_selection
import os


#Import the data
os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")
fd = pd.read_csv("Data\\Futures Data.csv", sep = ',')


#Convert all the columns of prices into returns
nat_gas = np.log(fd.iloc[:,1].shift(1)/fd.iloc[:,1])
nat_gas.pop(0)
ethanol = np.log(fd.iloc[:,2].shift(1)/fd.iloc[:,2])
ethanol.pop(0)
eurodollar = np.log(fd.iloc[:,3].shift(1)/fd.iloc[:,3])
eurodollar.pop(0)
oil = np.log(fd.iloc[:,4].shift(1)/fd.iloc[:,4])
oil.pop(0)
soybean = np.log(fd.iloc[:,5].shift(1)/fd.iloc[:,5])
soybean.pop(0)
brz_real = np.log(fd.iloc[:,6].shift(1)/fd.iloc[:,6])
brz_real.pop(0)
tyn = np.log(fd.iloc[:,7].shift(1)/fd.iloc[:,7])
tyn.pop(0)
vix = np.log(fd.iloc[:,8].shift(1)/fd.iloc[:,8])
vix.pop(0)
spy = np.log(fd.iloc[:,9].shift(1)/fd.iloc[:,9])
spy.pop(0)

#Stack the return vectors to create a returns matrix
X = np.column_stack((nat_gas, ethanol, eurodollar, oil, soybean, brz_real, tyn, vix))

#Create the return vector of our dependent variable
y = np.reshape(np.asarray(spy), (447,1))

#Run the PCA dimension reduction algorithm
pca = PCA(n_components = 2)
components = pca.fit_transform(X)
components_train, components_test, y_train, y_test = model_selection.train_test_split(components, y, test_size = 0.8, random_state = 7)

#Run the regression using the new components
model = LinearRegression()
model.fit(components_train, y_train)

#Print out the regression statistics
y_pred = model.predict(components_test)

print("\nThe R-squared from the PCA algorithm is", r2_score(y_test, y_pred))
print("The MSE from the PCA algorithm is", metrics.mean_squared_error(y_test,y_pred))
print("The MAE from the PCA algorithm is", metrics.mean_absolute_error(y_test,y_pred))

#Run the SVD dimension reduction algorithm
np.set_printoptions(precision = 2)

svd = TruncatedSVD(n_components = 2)
components = svd.fit_transform(X)
components_train, components_test, y_train, y_test = model_selection.train_test_split(components, y, test_size = 0.8, random_state = 7)

#Run the regression using the new components
model = LinearRegression()
model.fit(components_train, y_train)

#Print out the regression statistics
y_pred = model.predict(components_test)

print("\nThe R-squared from the SVD algorithm is", r2_score(y_test, y_pred))
print("The MSE from the SVD algorithm is", metrics.mean_squared_error(y_test,y_pred))
print("The MAE from the SVD algorithm is", metrics.mean_absolute_error(y_test,y_pred))

#Run the t-SNE dimension reduction algorithm
tsne = TSNE(n_components=2, perplexity= 10)
components = tsne.fit_transform(X)
components_train, components_test, y_train, y_test = model_selection.train_test_split(components, y, test_size = 0.8, random_state = 7)

#Run the regression using the new components
model = LinearRegression()
model.fit(components_train, y_train)

#Print out the regression statistics
y_pred = model.predict(components_test)

print("\nThe R-squared from the t-SNE algorithm is", r2_score(y_test, y_pred))
print("The MSE from the t-SNE algorithm is", metrics.mean_squared_error(y_test,y_pred))
print("The MAE from the t-SNE algorithm is", metrics.mean_absolute_error(y_test,y_pred))

#Run the ICA dimension reduction algorithm
ica = FastICA(n_components=2)
components = ica.fit_transform(X)
components_train, components_test, y_train, y_test = model_selection.train_test_split(components, y, test_size = 0.8, random_state = 7)

#Run the regression using the new components
model = LinearRegression()
model.fit(components_train, y_train)

#Print out the regression statistics
y_pred = model.predict(components_test)

print("\nThe R-squared from the ICA algorithm is", r2_score(y_test, y_pred))
print("The MSE from the ICA algorithm is", metrics.mean_squared_error(y_test,y_pred))
print("The MAE from the ICA algorithm is", metrics.mean_absolute_error(y_test,y_pred))
