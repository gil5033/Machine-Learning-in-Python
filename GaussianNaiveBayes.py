import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os

#Use which ever directory the data is in
os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")

#Load in the data
td = pd.read_csv( 'SP500 2000.csv', header = None )


#Convert to numpy arrays
X = np.delete( td.values, [0,8], axis = 1 )
y = np.delete( td.values, [0,1,2,3,4,5,6,7], axis = 1 ).ravel()

#Pop out the headers and create binary dependent variable
X = X[1:,:]
y = y[1:]
y = np.where(y == 'UP', 1, -1)

#Split the data using a 80/20 split
X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y, test_size = 0.20 )


#Fit the Gaussian Naive Bayes model
model = GaussianNB()
model.fit( X_train, y_train )
predictions = model.predict( X_test )


#Analyze the performance metrics
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test, predictions ) )
print( '\nACCURACY SCORE:\n' )
print( accuracy_score( y_test, predictions ) )
print( '\nCLASSIFICATION REPORT:\n' )
print( classification_report( y_test, predictions ) )

#Run a cross validation (10 folds)
kfold = model_selection.KFold( n_splits = 10 )
cv_results = model_selection.cross_val_score( model, X, y.ravel(), cv = kfold, scoring = 'accuracy' )

#Show the statistics from the cross validation
print( '\nMEAN AND STDEV:\n' )
print( cv_results.mean() )
print( cv_results.std() )
predicted_probas = model.predict_proba( X_test )