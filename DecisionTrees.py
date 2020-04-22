import pandas as pd
import numpy as np
import graphviz # conda install python-graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

#Create a pathway wherever your graphviz is found
os.environ["PATH"] += os.pathsep + 'C:/Users/Franster/anaconda3/Library/bin/graphviz/'

#use whichever directory you would like to store the graphics
os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")

#Store in the data
data = { 'Trend': [1, 1, 0, 0, 0], #1 stands for Pos and 0 for Neg
        'Volume': [1, 0, 1, 0, 1], #1 stands for High and 0 for Low
        'Time': [0, 1, 0, 1, 1],   #1 stands for PM and 0 for AM
        'Return': [1, 0, 1, 1, 0] }#1 stands for Up and 0 for Down

#Convert into pandas dataframe
data = pd.DataFrame(data = data)

#Convert into numpy array (unecessary but I already made it into a pandas df)
X = np.delete( data.values, 3, axis = 1 )
y = np.delete( data.values, [0,1,2], axis = 1 )

#Create the model
model = DecisionTreeClassifier( criterion = 'gini' )
model.fit( X, y )

#Create some graphics
dot_data = tree.export_graphviz( model, out_file = None )
graph = graphviz.Source( dot_data )
graph.render( 'DecisionTreeGraph' )

dot_data = tree.export_graphviz( model, out_file = None, 
                                feature_names = [ 'Trend', 'Volume', 'Time'], 
                                class_names = [ 'Up', 'Down'], 
                                filled = True, 
                                rounded = True, 
                                special_characters = True )
graph = graphviz.Source( dot_data )

# Performance evaluation
predictions = model.predict( X )
print( "\nAccuracy: ", accuracy_score( y, predictions ) * 100, "%" )


"""

The decision tree is defined as the following

First Rule: Is the time PM or AM
    If AM, then it will be a UP day
    If PM, continue to next decision rule

Second Rule: Is the trend Positive or Negative
    If Positive, then it will be DOWN day
    If Negative, continue to the next decision rule

Third Rule: Is the volume High or Low
    If Low, then it will be an UP day
    If High, then it will be a DOWN day
    
"""