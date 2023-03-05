"""
The preprocessing module contains functions for preprocessing data so that it is in a
format suitable for supervised learning
"""
# 3rd Party
import numpy as np
from sklearn.model_selection import train_test_split

def remove_extra_dimesions(X):
	"""
	If the observations are 2D, removes the z axis from each observation and
	removes the B_z and v_z predictor variables

	Parameters
	----------
	X : numpy.ndarray
		5D array representing the matrix of predictors

	Returns
	-------
	numpy.ndarray
		4D or 5D numpy array

			* The 0th axis specifies the observation
			* The 1st axis represents the x-direction
			* The 2nd axis represents the y-direction
			* If each observation is a 3D section of a disc: 

				* The 3rd axis represents the z-direction
				* The 4th axis represents the 8 different predictor variables

			* If each obseravtion is a 2D snapshot from the disc:
				* The 3rd axis represents the 6 different predictor variables
	"""
	if X.shape[3] == 1:
		# Remove the z axis, b_z
		X = np.squeeze(X, axis = 3)
		return np.delete(X, [2, 7], axis = 3)
	else:
		return X

def train_valid_test_split(X, y, train_size=0.8, valid_size=0.1):
	"""
	Split a data set into a training set, a validation set and a test set

	Parameters
	----------
	X : numpy.ndarray
		The design matrix
	y : numpy.ndarray
		The response variable
	train_size : float, default=0.8
		The proportion of the data set to put into the training set
	valid_size : float, default=0.1
		The proportion of the data set to put into the validation set

	Returns
	-------
	X_train : numpy.ndarray
	X_valid : numpy.ndarray
	X_test : numpy.ndarray
	y_train : numpy.ndarray
	y_valid : numpy.ndarray
	y_test : numpy.ndarray
	"""
	test_size = 1 - train_size - valid_size
	X_train, X_valid_test, y_train, y_valid_test = train_test_split(X,
		y,
		test_size=valid_size+test_size,
		stratify=y,
		random_state=42)
	X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test,
					y_valid_test,
					test_size = (test_size/(test_size+valid_size)),
					stratify = y_valid_test,
					random_state = 42)
	return X_train, X_valid, X_test, y_train, y_valid, y_test

def structurer(X, y, train_size=0.8, valid_size=0.1):
	"""
	Structure the data so it is ready for machine learning

	This function removes extra dimesnions from the matrix of predictors if the
	simulation is 2D. It then splits the data set into training, validation and
	test data sets.

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
	"""
	X = remove_extra_dimesions(X)
	(X_train, X_valid, X_test,
	y_train, y_valid, y_test) = train_valid_test_split(X, y, train_size, valid_size)
	return X_train, X_valid, X_test, y_train, y_valid, y_test
