import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Change directory to find the data
import os
os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")

#Load in the data set
td = pd.read_csv("trade_data.csv")

#Create independent and dependent vairable data
X = np.delete( td.values, [0,1], axis = 1 )
y = np.delete( td.values, [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                           20,21,22,23,24,25,26,27,28,29,30], axis = 1 )

#Change dependent to a binary value to predict up or down
y = np.where(y[1:]>np.delete(y,len(y)-1, 0), 1, 0)
#Adjust size on X to remove the latest datum
X = np.delete( X, [len(X)-1], axis = 0)
#Lastly, standardize the independent vairable
scalar = StandardScaler()
X = scalar.fit_transform(X)

#Split the data for testing with a shuffle
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, shuffle = True)

#Create the neural network model
model = Sequential()

#Create the layers in the model
model.add( Dense( 4, kernel_initializer = 'random_uniform', bias_initializer = 'zeros',
activation = 'sigmoid', input_dim = X.shape[ 1 ])) # Input layer
model.add( Dense( 5, activation = 'sigmoid' ) ) #hidden layer
model.add( Dense( 6, activation = 'sigmoid' ) ) #hidden layer
model.add( Dense( 4, activation = 'sigmoid' ) ) #hidden layer
model.add( Dense( 3, activation = 'sigmoid' ) ) #hidden layer
model.add( Dense( 1, activation = 'sigmoid' ) ) #Output Layer

#Compile the model and run 1000 epochs
model.compile( optimizer = 'adam', loss = 'mean_squared_error', metrics = [ 'accuracy' ] )
model.fit( X_train, y_train, epochs = 1000, verbose = False )

#Lets check the performance
predictions = model.predict( X_test )
print( '\nRAW PREDICTIONS:\n' )
print( predictions )

predictions = np.round( predictions )

print( '\n CATEGORICAL PREDICTIONS:\n' )
print( predictions )
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test, predictions ) )

print( '\nACCURACY SCORE:\n' )
print(accuracy_score(y_test, predictions)*100, "%")