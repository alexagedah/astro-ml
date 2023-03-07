"""
The preprocessor module contains the functions for preprocessing data so that it
is in a format suitable for supervised learning
"""
# 3rd Party
import numpy as np
import tensorflow as tf
# Local
from . import dimension_reduction, features, normalisation, split

def preprocessor(X, y, train_size=0.8, valid_size=0.1, feature_list=[], apply_standard_scaling="yes"):
	"""
	Preprocess the data so it is ready for machine learning

	Preprocessing steps

		#. Add additional features (optional)
		#. Remove extra dimensions from the matrix of predictors if the
		   simulation is 2D
		#. Split the data set into training, validation and test data sets
		#. Apply standard scaling to the predictors (optional)
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
	feature_list :
		A list of the additional features to include in the model
	apply_standard_scaling : 
		Whether standard scaling should be applied to the predictors

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
	feature_map : dictionary of {int:str}
		Dictionary indicating which features are at which index in the matrix of
		predictors
	min_max_transformer : sklearn.preprocessing.MinMaxScaler
		Scikit-learn transformer for the response
	"""
	X, feature_map = features.add_features(X, feature_list)
	X, feature_map = dimension_reduction.remove_extra_dimesions(X, feature_map)
	(X_train, X_valid, X_test,
	y_train, y_valid, y_test) = split.train_valid_test_split(X, y.reshape(-1, 1), train_size, valid_size)
	if apply_standard_scaling == "yes":
		X_train, X_valid, X_test = normalisation.standard_scaler(X_train, X_valid, X_test)
	y_train, y_valid, y_test, min_max_transformer = normalisation.min_max_scaler(y_train, y_valid, y_test)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, feature_map, min_max_transformer

