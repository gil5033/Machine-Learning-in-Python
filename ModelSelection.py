import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model as lm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from prunefunction import prune_function
import operator
import os

#Change the directory to grab wherever the data is
os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")

#Load the data into a pndas dataframe
td = pd.read_csv("credit_approval.csv")

#Clean out the data
td = td.replace('?', None)
td = td.dropna( how = 'all' )
#Convert into numpy arrays
X = np.delete(td.values, 15, axis = 1)
y = np.delete(td.values, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], axis = 1)

#Find columns that have char and string values
Xs = [ 0,3,4,5,6,8,9,11,12 ]

#label encode them into binary values
le = preprocessing.LabelEncoder()
for i in Xs:
    X[ :, i ] = le.fit_transform( X[ :, i ] )
#Convert the dependent variable into binary values
y = le.fit_transform( y.ravel() )


#Run the LDA classification
X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y.ravel(), 
                                                                    test_size = .75,
                                                                    shuffle = True)
model = LinearDiscriminantAnalysis()
model.fit( X_test, y_test )
predictions = model.predict( X_test )
print('\n\nLINEAR DISCRIMINANT ANALYSIS')
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test, predictions ) )
print( '\nACCURACY SCORE:\n' )
print( accuracy_score( y_test, predictions ) ) 


#Run the QDA classification
X_train2, X_test2, y_train2, y_test2 = model_selection.train_test_split( X, y.ravel(), 
                                                                    test_size = .75,
                                                                    shuffle = True)
model2 = QuadraticDiscriminantAnalysis()
model2.fit( X_test2, y_test2 )
predictions2 = model2.predict( X_test2 )
print('\n\nQUADRATIC DISCRIMINANT ANALYSIS')
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test2, predictions2 ) )
print( '\nACCURACY SCORE:\n' )
print( accuracy_score( y_test2, predictions2 ) ) 


#Run the logistic regression
X_train3, X_test3, y_train3, y_test3 = model_selection.train_test_split( X, y.ravel(), 
                                                                    test_size = .75,
                                                                    shuffle = True)
model3 = lm.LogisticRegression()
model3.fit( X_test3, y_test3 )
predictions3 = model3.predict( X_test3 )
print('\n\nLOGISTIC REGRESSION')
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test3, predictions3 ) )
print( '\nACCURACY SCORE:\n' )
print( accuracy_score( y_test3, predictions3 ) ) 


#Run the Naive Bayes classification
X_train4, X_test4, y_train4, y_test4 = model_selection.train_test_split( X, y.ravel(), 
                                                                    test_size = .75,
                                                                    shuffle = True)
model4 = GaussianNB()
model4.fit( X_test4, y_test4 )
predictions4 = model4.predict( X_test4 )
print('\n\nNAIVE BAYES')
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test4, predictions4 ) )
print( '\nACCURACY SCORE:\n' )
print( accuracy_score( y_test4, predictions4 ) )


#Run the decision tree classification
X_train5, X_test5, y_train5, y_test5 = model_selection.train_test_split( X, y.ravel(), 
                                                                    test_size = .75,
                                                                    shuffle = True)
model5 = DecisionTreeClassifier(criterion = 'gini')
model5.fit( X_test5, y_test5 )
prune_function( model5.tree_, 0, 35 )
predictions5 = model5.predict( X_test5 )
print('\n\nDECISION TREE')
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test5, predictions5 ) )
print( '\nACCURACY SCORE:\n' )
print( accuracy_score( y_test5, predictions5 ) )


#Run the support vector machines
X_train6, X_test6, y_train6, y_test6 = model_selection.train_test_split( X, y.ravel(), 
                                                                    test_size = .75,
                                                                    shuffle = True)
model6 = svm.SVC(kernel = 'rbf')
model6.fit( X_train6, y_train6 )
predictions6 = model6.predict( X_test6 )
print('\n\nSUPPORT VECTOR MACHINES')
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test6, predictions6 ) )
print( '\nACCURACY SCORE:\n' )
print( accuracy_score( y_test6, predictions6 ) )


accuracy_scores = {'LDA': accuracy_score( y_test, predictions ), 
                   'QDA': accuracy_score( y_test2, predictions2 ),
                   'Logistic Regression': accuracy_score( y_test3, predictions3),
                   'Naive Bayes': accuracy_score( y_test4, predictions4 ),
                   'Decision Tree': accuracy_score( y_test5, predictions5 ),
                   'Support Vector Machines': accuracy_score( y_test6, predictions6 )}

print('\n')
print(max(accuracy_scores.items(), key=operator.itemgetter(1))[0], "is the best model for classifying")
print('\n')