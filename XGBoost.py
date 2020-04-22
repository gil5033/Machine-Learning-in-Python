import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import os

os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")

test_data = pd.read_csv("trinary_test_data.csv", header = None)

X = np.delete(test_data.values,4, axis = 1)
y = np.delete(test_data.values, [0,1,2,3], axis = 1)

params = { 'max_depth' : range( 2, 10, 2 ),
 'min_child_weight' : range( 1, 6, 2 ) }

gscv = GridSearchCV( estimator = XGBClassifier( learning_rate = 0.1, n_estimators = 140,
                        max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8,
                        colsample_bytree = 0.8, objective = 'multi:softprob', nthread = 1,
                        scale_pos_weight = 1, seed = 27 ),
                        param_grid = params, n_jobs = 1, iid = False, cv = 5 )

gscv.fit( X, y.ravel() )

print( '\nGRID SCORES:\n' )
print( gscv.grid_scores_ )

print( '\nBEST PARAMETERS:\n' )
print( gscv.best_params_ )

print( '\nBEST SCORE:\n' )
print( gscv.best_score_ )
