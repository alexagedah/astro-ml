"""
The preprocessor module contains the functions for preprocessing data so that it
is in a format suitable for supervised learning
"""
# 3rd Party
import numpy as np
import tensorflow as tf
# Local
from . import normalisation, split

def preprocessor(X, y, feature_scaling="standard", response_scaling="min-max", train_size=0.8, valid_size=0.1):
	"""
	Preprocess the data so it is ready for machine learning

	Preprocessing steps

		#. Split the data set into training, validation and test data sets
		#. Apply standard scaling to the predictors (optional)
		#. Apply min-max scaling to the response so that it is between -1 and 1
		   (optional)

	Parameters
	----------
	X : numpy.ndarray
		5D array representing the matrix of predictors
	y : numpy.ndarray
		1D array representing the vector of responses
	feature_scaling : default="standard"
		The scaling that should be applied to the features
	response_scaling : default="min-max"
		The scaling that should be applied to the responses
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
	response_transformer : sklearn.preprocessing.Transfomer or None
		A Scikit-learn transformer for the response. If no transformer is used,
		return None
	"""
	(X_train, X_valid, X_test,
	y_train, y_valid, y_test) = split.train_valid_test_split(X, y.reshape(-1, 1), train_size, valid_size)
	if feature_scaling == "standard":
		X_train, X_valid, X_test = normalisation.standard_scaler(X_train, X_valid, X_test)
	if response_scaling == "min-max":
		y_train, y_valid, y_test, response_transformer = normalisation.min_max_scaler(y_train, y_valid, y_test)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, response_transformer

