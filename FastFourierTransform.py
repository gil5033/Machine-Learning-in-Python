import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

#Use whichever directory the data is located in
os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")

#Load in the data set
house = pd.read_csv("PAYNSA.csv")

#Save its length
x = len(house)

#Convert to a numpy array
T = np.arange( 0, 1, 1/x )
y = ( np.delete( house.values, [ 0 ], axis = 1 ) ).astype( float )

#Take a look at the data
plt.plot( y )
plt.title( 'Data By Month' )
plt.show()

#Set up models for linear regression
model = LinearRegression()
model.fit( T[:,np.newaxis], y )

#Gain some new_ys from the linear regression
predictions = model.predict( T[:,np.newaxis] )

#Detrend the dependent variable
detrended_y = ( y - predictions ).ravel()
plt.plot( detrended_y )
plt.title( 'Detrended Y-values From Linear Regression' )
plt.show()

#Lets run it!
result = np.fft.fft( detrended_y )

#Get rid of the complex conjugates
#We only need half of them, the real number side
half_n = int( x / 2 )
result = result[ range( half_n ) ]

#Find the parameters to our trig functions
freqs = np.arange( half_n )
As = np.abs( result ) / half_n
phis = np.angle( result ) * 180 / np.pi + 90
plt.plot( freqs, As )
plt.title( 'Frequency Domain' )
plt.show()

#Pull out the top frequencies, and their amplitudes and phase shifts
#The tops can be changed, it is currently at 250
tops = np.where( As > 250, 1, 0 )
top_As = tops * As
top_phis = tops * phis

#Initialize an array to use for new_ys
j = 0
new_y = np.zeros( x )

#Formulate back an equation for time domain
for i in tops:
 new_y += top_As[ j ] * np.sin( 2 * np.pi * ( j * int( i ) ) * T + top_phis[ j ] * ( np.pi / 180 ) )
 j += 1
 
#Plot the model we have for detrended y
plt.plot( detrended_y )
plt.plot( new_y )
plt.title( 'Frequency Model' )
plt.show()

#Plot the residuals, should be normal
residuals = detrended_y.ravel() - new_y
plt.plot( residuals )
plt.title( 'Residuals' )
plt.show()
plt.hist( residuals, bins = 20 )
plt.title( 'Residuals Histogram' ) #Should appear to be normal
plt.show()