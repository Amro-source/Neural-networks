# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:27:48 2020

@author: Zikantika
"""


import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import model_from_json
import numpy
import os
# load dataset
#dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataframe = pandas.read_csv("housing.csv")

dataset = dataframe.values
print(dataset)

A=dataset.shape
print(A)
# split into input (X) and output (Y) variables
X = dataset[:,0:13]

#X = dataset[:,0:10]

print(X)
print(X.shape)
Y = dataset[:,13]
print(Y)
print(Y.shape)

# define base model
#def baseline_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
#	model.add(Dense(1, kernel_initializer='normal'))
#	# Compile model
#	model.compile(loss='mean_squared_error', optimizer='adam')
#    model.summary()
#	return model


def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal',activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
     json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
    return model
#
#model = Sequential()
#model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
#model.add(Dense(10, kernel_initializer='normal',activation='relu'))
#model.add(Dense(1, kernel_initializer='normal'))
## Compile model
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.summary()

kfold = KFold(n_splits=10)

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)
#estimator = KerasRegressor(model, epochs=100, batch_size=5, verbose=1)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#model=baseline_model
#print(model)
#model.summary()
# serialize model to JSON
#model_json = model.to_json()
##with open("model.json", "w") as json_file:
##    json_file.write(model_json)
### serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")
# Export the model to a SavedModel
#model.save('housemodelffnn.h5', save_format='tf')
#config = model.get_config()
#weights = model.get_weights()


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))