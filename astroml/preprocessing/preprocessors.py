"""
The preprocessors module contains functions for preprocessing data so that it is
in a format suitable for supervised learning
"""

# 3rd Party
import numpy as np
import tensorflow as tf
# Local
from . import dimension_reduction, features, normalisation, split

def simple_preprocessor(X, y, train_size, valid_size, _=None):
	"""
	Preprocess the data so it is ready for machine learning

	Preprocessing steps

		#. Remove extra dimensions from the matrix of predictors if the
		   simulation is 2D
		#. Split the data set into training, validation and test data sets
		#. Apply min-max scaling to the response so that it is between -1 and 1

	This preprocessor is considered "simple" since the only preprocessing that
	is done is just transforming the response

	Parameters
	----------
	X : numpy.ndarray
		5D array representing the matrix of predictors
	y : numpy.ndarray
		1D array representing the vector of responses
	train_size : float
		The proportion of the data set to put into the training data set
	valid_size : float
		The proportion of the data set to put into the validation data set

	Returns
	-------
	X_train : numpy.ndarray
		4D (if the simulation is 2D) or 5D (if the simulation is 3D)
		numpy.ndarray representing the matrix of predictors for the training
		data set
	X_valid : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		validation data set
	X_test : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		test data set
	y_train : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the training
		data set
	y_valid : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the validation
		data set
	y_test : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the test
		data set
	min_max_transformer : sklearn.preprocessing.MinMaxScaler
		The transformer used to transform the response variable
	"""
	X = dimension_reduction.remove_extra_dimesions(X)
	(X_train, X_valid, X_test,
	y_train, y_valid, y_test) = split.train_valid_test_split(X, y, train_size, valid_size)
	y_train, y_valid, y_test, min_max_transformer = normalisation.min_max_scaler(y_train, y_valid, y_test)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer

def standard_preprocessor(X, y, train_size, valid_size, features_list=[]):
	"""
	Preprocess the data so it is ready for machine learning

	Preprocessing steps

		#. Add additional features
		#. Remove extra dimensions from the matrix of predictors if the
		   simulation is 2D
		#. Split the data set into training, validation and test data sets
		#. Apply standard scaling to the predictors
		#. Apply min-max scaling to the response so that it is between -1 and 1

	This preprocessor is considerd "standard' as these are the typical steps 
	that would get taken in a machine learning project

	Parameters
	----------
	X : numpy.ndarray
		5D array representing the matrix of predictors
	y : numpy.ndarray
		1D array representing the vector of responses
	train_size : float, default=0.8
		The proportion of the data set to put into the training data set
	valid_size : float, default=0.1
		The proportion of the data set to put into the validation data set

	Returns
	-------
	X_train : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		training data set
	X_valid : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		validation data set
	X_test : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		test data set
	y_train : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the training
		data set
	y_valid : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the validation
		data set
	y_test : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the test
		data set
	min_max_transformer : sklearn.preprocessing.MinMaxScaler
	"""
	X = features.add_features(X, features_list)
	X = dimension_reduction.remove_extra_dimesions(X)
	(X_train, X_valid, X_test,
	y_train, y_valid, y_test) = split.train_valid_test_split(X, y, train_size, valid_size)
	X_train, X_valid, X_test = normalisation.standard_scaler(X_train, X_valid, X_test)
	y_train, y_valid, y_test, min_max_transformer = normalisation.min_max_scaler(y_train, y_valid, y_test)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer

def preprocessor(X, y, preprocessor_type="smart", features_list=[], train_size=0.8, valid_size=0.1):
	"""
	Preprocess the data so it is ready for machine learning

	Parameters
	----------
	X : numpy.ndarray
		5D array representing the matrix of predictors
	y : numpy.ndarray
		1D array representing the vector of responses
	features_list : list of str
		A list of the features to add
	preprocessor_type : str, default="physical"
		The type of preprocessor to use. options are "simple","standard"
	train_size : float, default=0.8
		The proportion of the data set to put into the training data set
	valid_size : float, default=0.1
		The proportion of the data set to put into the validation data set

	Returns
	-------
	X_train : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		training data set

			* The 0th axis specifies the observation
			* The 1st axis represents the x-direction
			* The 2nd axis represents the y-direction
			* If each observation is a 3D section of a disc: 

				* The 3rd axis represents the z-direction
				* The 4th axis represents the 8 different predictor variables

			* If each obseravtion is a 2D snapshot from the disc:
				* The 3rd axis represents the 6 different predictor variables
				* There is no 4th axis since the numpy.ndarray is 4D.
	X_valid : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		validation data set
	X_test : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		test data set
	y_train : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the training
		data set
	y_valid : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the validation
		data set
	y_test : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the test
		data set
	min_max_transformer : sklearn.preprocessing.MinMaxScaler
	"""
	if preprocessor_type == "simple":
		function = simple_preprocessor
	elif preprocessor_type == "standard":
		function = standard_preprocessor
	elif preprocessor_type == "smart":
		function = smart_preprocessor
	X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer = function(X,
		y.reshape(-1,1),
		train_size,
		valid_size, features_list)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer










