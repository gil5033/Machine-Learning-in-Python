import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import talib
import pandas_datareader as web
from datetime import datetime

#Use pandas datareader because we are lazy (aka smart)
start = datetime(2017, 12, 1)
end = datetime(2019, 11, 1)
source = 'yahoo'
Stock = 'SPY'

#Load in the data set from datasource
td = web.DataReader(Stock, source, start, end)

#Only take the open, high, low, and close
td = td[['Open', 'High', 'Low', 'Close']]

#Create some technical indicators
td[ 'H-L' ] = td[ 'High' ] - td[ 'Low' ]
td[ 'C-O' ] = td[ 'Close' ] - td[ 'Open' ]
td[ '3day MA' ] = td[ 'Close' ].shift( 1 ).rolling( window = 3 ).mean()
td[ '10day MA' ] = td[ 'Close' ].shift( 1 ).rolling( window = 10 ).mean()
td[ '30day MA' ] = td[ 'Close' ].shift( 1 ).rolling( window = 30 ).mean()
td[ 'Std_dev' ] = td[ 'Close' ].rolling( 5 ).std()
td[ 'RSI' ] = talib.RSI( td[ 'Close' ].values, timeperiod = 9 )
td[ 'Williams %R' ] = talib.WILLR( td[ 'High' ].values, td[ 'Low' ].values,
td[ 'Close' ].values, 7 )
td[ 'Price_Rise' ] = np.where( td[ 'Close' ].shift( -1 ) > td[ 'Close' ], 1, 0 )

#Drop the NA rows that arise from the 30day MA and others
td = td.dropna()

#Convert the values into a numpy array
td = td.values
X = td[ :, 0:12 ].astype( float )
y = td[ :, 12 ]

#Standardize the independent variable
scaler = StandardScaler()
X = scaler.fit_transform( X )

#Split the data into training and testing data with shuffle
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, shuffle = True)

#Create the neural network
model = Sequential()

#Create the layers in the model
model.add( Dense( 4, kernel_initializer = 'random_uniform', bias_initializer = 'zeros',
activation = 'linear', input_dim = X.shape[ 1 ])) # Input layer
model.add( Dense( 3, activation = 'linear' ) ) #hidden layer
model.add( Dense( 1, activation = 'sigmoid' ) ) #Output Layer

#Compile the model and run 1000 epochs
model.compile( optimizer = 'adam', loss = 'mean_squared_error', metrics = [ 'accuracy' ] )
model.fit( X_train, y_train, epochs = 1000, verbose = False )

#Lets check the performance
predictions = model.predict( X_test )
predictions = np.round(predictions)

print( '\n CLASSIFICATION REPORT:\n' )
print( classification_report(y_test, predictions ) )
print( '\nCONFUSION MATRIX:\n' )
print( confusion_matrix( y_test, predictions ) )

print( '\nACCURACY SCORE:\n' )
print(accuracy_score(y_test, predictions)*100, "%")
