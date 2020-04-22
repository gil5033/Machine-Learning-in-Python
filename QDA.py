import numpy as np
import pandas as pd
from datetime import datetime
from talib import MA_Type
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
import talib as ta

start = datetime(2015, 1, 1)
end = datetime(2018, 7, 31)

data = web.DataReader( 'GOOG', 'yahoo', start, end )

stock_returns = np.log(data['Adj Close']/data['Adj Close'].shift(1))
sig = np.std(stock_returns)

data['Moving Average - 10'] = ta.SMA(np.array(data['Adj Close']), timeperiod = 10)
data['Upper'], data['Middle'], data['Lower'] = ta.BBANDS(np.array(data['Adj Close']), matype = MA_Type.T3)
data['Exponential Moving Average'] = ta.EMA(np.array(data['Adj Close']))
data['Moving Average - 25'] = ta.SMA(np.array(data['Adj Close']), timeperiod = 25)

data = data.iloc[29:]

stock_returns = np.log(data['Adj Close']/data['Adj Close'].shift(1))
sig = np.std(stock_returns)

X = np.delete( data.values,[0,1,2,3,4,5], axis = 1)
y = np.where(stock_returns > (sig/2), 1, np.where(stock_returns < (-sig)/2, -1, 0))


X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y.ravel(), test_size = 0.20, shuffle = False )


model = QuadraticDiscriminantAnalysis()
model.fit(X_train,y_train)

probability = model.predict_proba( X_test )

pred_y = model.predict( X_test )

print( '\nConfusion matrix:\n' )
print( metrics.confusion_matrix( y_test, pred_y ) )
print( '\nClassification report:\n' )
print( metrics.classification_report( y_test, pred_y ) )
print( 'Model score:\n' )
print( model.score( X_test, y_test ).round( 2 ) )


data['Returns'] = np.log(data['Adj Close']/data['Adj Close'].shift(1))
data['Predicted Signal'] = model.predict(X)

data = data.iloc[1:]

cum_returns = np.cumsum(data[:]['Returns'])

data['Strategy Returns'] = data['Returns'] * data['Predicted Signal'].shift( 1 )

cum_strategy_returns = np.cumsum( data[ : ][ 'Strategy Returns' ] )

plt.figure( figsize = ( 10, 5 ) )
plt.plot( cum_returns, color = 'r',label = 'Stock Returns' )
plt.plot( cum_strategy_returns, color= 'g', label = 'Strategy Returns' )
plt.legend()
plt.show()