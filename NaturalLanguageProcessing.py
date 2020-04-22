import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split


import os
os.chdir("C:\\Users\\Franster\\Desktop\\Computer Science\\Code\\Python Code\\Data")

df = pd.read_csv( 'TSLA_tweets.txt' , header = None, sep = '\t')
reviews = df[ 0 ].tolist()

reviews_clean = np.empty( len( reviews ), dtype = object )
for i in range( len( reviews ) ):
    
    tokens = reviews[ i ].split()
    # Get rid of punctuations
    table = str.maketrans('', '', punctuation)
    
    tokens = [ w.translate( table ) for w in tokens ]
    tokens = [ word for word in tokens if word.isalpha() ]
    
    # Get rid of stop words
    stop_words = set( stopwords.words( 'english' ) )
    tokens = [ w for w in tokens if not w in stop_words ]
    
    
    # Get rid of short words
    tokens = [ word for word in tokens if len( word ) > 1 ]
    tokens = ' '.join(tokens)
    reviews_clean[ i ] = tokens[:]
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts( reviews_clean )

max_length = max( [ len( s.split() ) for s in reviews_clean ] )
vocab_size = len( tokenizer.word_index ) + 1
encoded = tokenizer.texts_to_sequences( reviews_clean )

# Pad the encoded sequences to get X
X = pad_sequences( encoded, maxlen = max_length, padding = 'post' )
y = df[ 1 ].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, shuffle = True )

# Build the convolutional neural network

# Channel 1
inputs1 = Input( shape = ( 17, ) )
embedding1 = Embedding( vocab_size, 100 )( inputs1 )
conv1 = Conv1D( filters = 32, kernel_size = 4, activation = 'relu' )( embedding1 )
drop1 = Dropout( 0.5 )( conv1 )
pool1 = MaxPooling1D( pool_size = 2 )( drop1 )
flat1 = Flatten()( pool1 )

# Channel 2
inputs2 = Input( shape = ( 17, ) )
embedding2 = Embedding( vocab_size, 100 )( inputs2 )
conv2 = Conv1D( filters = 32, kernel_size = 6, activation = 'sigmoid' )( embedding2 )
drop2 = Dropout( 0.5 )( conv2 )
pool2 = MaxPooling1D( pool_size = 2 )( drop2 )
flat2 = Flatten()( pool2 )

# Channel 3
inputs3 = Input( shape = ( 17, ) )
embedding3 = Embedding( vocab_size, 100 )( inputs3 )
conv3 = Conv1D( filters=32, kernel_size = 8, activation = 'sigmoid' )( embedding3 )
drop3 = Dropout( 0.5 )( conv3 )
pool3 = MaxPooling1D( pool_size = 2 )( drop3 )
flat3 = Flatten()( pool3 )

# Combine the channels
combined = concatenate( [ flat1, flat2, flat3 ] )

# Create the dense layers
dense1 = Dense( 10, activation = 'sigmoid' )( combined )
outputs = Dense( 1, activation = 'sigmoid' )( dense1 )
model = Model( inputs = [ inputs1, inputs2, inputs3 ], outputs = outputs )

# Compile and fit the CNN
model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = [ 'accuracy' ] )
model.fit( [ X_train, X_train, X_train ], y_train, epochs = 100, batch_size= 16 )

# Evaluate the model on training and test data sets
loss, acc = model.evaluate( [ X_train, X_train, X_train ], y_train, verbose = 0 )
print( 'TRAINING ACCURACY: %.2f' % acc )

loss, acc = model.evaluate( [ X_test, X_test, X_test ], y_test, verbose = 0 )
print( '\nTEST ACCURACY: %.2f' % acc )
