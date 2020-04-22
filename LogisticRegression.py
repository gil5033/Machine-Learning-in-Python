import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import talib as ta
from talib import MA_Type
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn import metrics
from sklearn import model_selection
import numpy as np

start = datetime(2015, 1, 1)
end = datetime(2018, 7, 31)

data = web.DataReader( 'GOOG', 'yahoo', start, end )

data['Moving Average - 10'] = ta.SMA(np.array(data['Adj Close']), timeperiod = 10)
data['Upper'], data['Middle'], data['Lower'] = ta.BBANDS(np.array(data['Adj Close']), matype = MA_Type.T3)
data['Exponential Moving Average'] = ta.EMA(np.array(data['Adj Close']))
data['Moving Average - 25'] = ta.SMA(np.array(data['Adj Close']), timeperiod = 25)

data = data.iloc[29:]

y = np.where( ( data['Adj Close'].shift( -1 ) < data['Adj Close'] ), -1, 1)
X = np.delete( data.values,[0,1,2,3,4,5], axis = 1)

X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y.ravel(), test_size = 0.20, shuffle = False )

model = lm.LogisticRegression()
model = model.fit( X_train, y_train )

probability = model.predict_proba( X_test )

pred_y = model.predict( X_test )

print( '\nConfusion matrix:\n' )
print( metrics.confusion_matrix( y_test, pred_y ) )
print( '\nClassification report:\n' )
print( metrics.classification_report( y_test, pred_y ) )
print( 'Model score:\n' )
print( model.score( X_test, y_test ).round( 2 ) )

data['Predicted Signal'] = model.predict(X)
data['Returns'] = np.log(data['Adj Close']/data['Adj Close'].shift(1))

data = data.iloc[1:]

cum_google_returns = np.cumsum( data[ : ][ 'Returns' ] )

data['Startegy Returns'] = data['Returns'] * data['Predicted Signal'].shift( 1 )

cum_strategy_returns = np.cumsum( data[ : ][ 'Startegy Returns' ] )

plt.figure( figsize = ( 10, 5 ) )
plt.plot( cum_google_returns, color = 'r',label = 'Google Returns' )
plt.plot( cum_strategy_returns, color= 'g', label = 'Strategy Returns' )
plt.legend()
plt.show()