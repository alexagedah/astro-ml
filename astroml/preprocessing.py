"""
The preprocessing module contains functions for preprocessing data so that it is in a
format suitable for supervised learning
"""
# 3rd Party
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

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

def train_valid_test_split(X, y, train_size, valid_size):
	"""
	Split a data set into a training set, a validation set and a test set

	Parameters
	----------
	X : numpy.ndarray
		The design matrix
	y : numpy.ndarray
		The response variable
	train_size : float
		The proportion of the data set to put into the training set
	valid_size : float
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

def full_standard_scaler(X_train, X_valid, X_test):
	"""
	Standardise all the predictors

	Parameters
	---------
	X_train : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		training data set
	X_valid : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		validation data set
	X_test : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
		test data set
	Returns
	-------
	X_train_scaled : numpy.ndarray
	X_valid_scaled : numpy.ndarray
	X_test_scaled : numpy.ndarray

	norm_layer : tf.keras.layers.Normalization
		The fitted normalisation layer
	"""
	norm_layer = tf.keras.layers.Normalization()
	norm_layer.adapt(X_train)
	X_train_scaled = norm_layer(X_train)
	X_valid_scaled = norm_layer(X_valid)
	X_test_scaled = norm_layer(X_test)
	return X_train_scaled, X_valid_scaled, X_test_scaled

def min_max_scaler(y_train, y_valid, y_test):
	"""
	Apply min-max scaling to the vector of response

	Parameters
	---------
	y_train : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the training
		data set
	y_valid : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the validation
		data set
	y_test : numpy.ndarray
		1D numpy.ndarray representing the vector of response for the test
		data set

	Returns
	-------
	y_train_scaled : numpy.ndarray
	y_valid_scaled : numpy.ndarray
	y_test_scaled : numpy.ndarray
	min_max_transformer : sklearn.preprocessing.MinMaxScaler
		The transformer for min-max scaling
	"""
	min_max_transformer = MinMaxScaler(feature_range=(-1,1))
	y_train_scaled = min_max_transformer.fit_transform(y_train)
	y_valid_scaled = min_max_transformer.transform(y_valid)
	y_test_scaled = min_max_transformer.transform(y_test)
	return y_train_scaled, y_valid_scaled, y_test_scaled, min_max_transformer

def add_velocity_magnitude(X):
	"""
	Add the magnitude of the velocity to the matrix of predictors

	Parameters
	----------
	X : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
	training data set

	Returns
	-------
	X_transformed : numpy.ndarray
		4D or 5D numpy array representing the matrix of predictors with the magnitude
	of the magnetic field as a predictor.
	"""
	velocity_magnitude = np.sqrt(X[:,:,:,:,5]**2 + X[:,:,:,:,6]**2 + X[:,:,:,:,7]**2)
	X_transformed = np.concatenate([X, velocity_magnitude[:,:,:,:,np.newaxis]], 4)
	for i in [5, 6, 7]:
		X_transformed[:,:,:,:,i] = np.divide(X_transformed[:,:,:,:,i],
			velocity_magnitude,
			out=np.zeros_like(X_transformed[:,:,:,:,i]),
			where=velocity_magnitude!=0)
	return X_transformed

def add_magnetic_field_magnitude(X):
	"""
	Add the magnitude of the magnetic field to the matrix of predictors and norma

	Parameters
	----------
	X : numpy.ndarray
		4D or 5D numpy.ndarray representing the matrix of predictors for the
	training data set

	Returns
	-------
	X_transformed : numpy.ndarray
		4D or 5D numpy array representing the matrix of predictors with the magnitude
	of the magnetic field as a predictor.
	"""
	magnetic_field_magnitude = np.sqrt(X[:,:,:,:,0]**2 + X[:,:,:,:,1]**2 + X[:,:,:,:,2]**2)
	X_transformed = np.concatenate([X, magnetic_field_magnitude[:,:,:,:,np.newaxis]], 4)
	for i in [0, 1, 2]:
		X_transformed[:,:,:,:,i] = np.divide(X_transformed[:,:,:,:,i],
			magnetic_field_magnitude,
			out=np.zeros_like(X_transformed[:,:,:,:,i]),
			where=magnetic_field_magnitude!=0)
	return X_transformed

def standard_scaler(x, mean, std):
	return (x - mean)/std

def smart_standard_scaler(X_train, X_valid, X_test, variable_number):
	"""
	Apply standard scaling to a variable spefied by the number 
	"""
	std = X_train[:,:,:,:,variable_number].std()
	mean = X_train[:,:,:,:,variable_number].mean()
	X_train[:,:,:,:,variable_number] = standard_scaler(X_train[:,:,:,:,variable_number], mean, std)
	X_valid[:,:,:,:,variable_number] = standard_scaler(X_valid[:,:,:,:,variable_number], mean, std)
	X_test[:,:,:,:,variable_number] = standard_scaler(X_test[:,:,:,:,variable_number], mean, std)
	return X_train, X_valid, X_test

def dumb_preprocessor(X, y, train_size=0.8, valid_size=0.1):
	"""
	Preprocess the data so it is ready for machine learning

	Preprocessing steps

		#. Remove extra dimensions from the matrix of predictors if the
		simulation is 2D
		#. Split the data set into training, validation and test data sets

	This preprocessor is considered "dumb" since there is esentially no
	preprocessing

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
	min_max_transformer : sklearn.preprocessing.MinMaxScaler
	"""
	X = remove_extra_dimesions(X)
	(X_train, X_valid, X_test,
	y_train, y_valid, y_test) = train_valid_test_split(X, y, train_size, valid_size)
	y_train, y_valid, y_test, min_max_transformer = min_max_scaler(y_train, y_valid, y_test)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer

def physical_preprocessor(X, y, train_size=0.8, valid_size=0.1):
	"""
	Preprocess the data so it is ready for machine learning

	Preprocessing steps

		#. Remove extra dimensions from the matrix of predictors if the
		simulation is 2D
		#. Add the magnitude of the magnetic field as a predictor and normalise
		the components of the magnetic field
		#. Add the magnitude of the velocity as a predictor and normalise
		the components of the velocity
		#. Split the data set into training, validation and test data sets
		#. Apply standard scaling to the magnitude of the magnetic field, gas
		pressure, density and magnitude of the velocity
		#. Apply min-max scaling to the response so that it is between -1 and 1

	This preprocessor is considered "physical" since it transforms the variables
	in a way which makes sense physically.

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
	min_max_transformer : sklearn.preprocessing.MinMaxScaler
	"""
	X_transformed_one = add_magnetic_field_magnitude(X)
	X_transformed_two = add_velocity_magnitude(X_transformed_one)
	(X_train, X_valid, X_test,
	y_train, y_valid, y_test) = train_valid_test_split(X_transformed_two, y, train_size, valid_size)
	for key, value in {"Gas Pressure":3,"Density":4,"Magnetic Field Magnitude":8,"Velocity Magnitude":9}.items():
		X_train, X_valid, X_test = smart_standard_scaler(X_train, X_valid, X_test, value)
	X_train = remove_extra_dimesions(X_train)
	X_valid = remove_extra_dimesions(X_valid)
	X_test = remove_extra_dimesions(X_test)
	y_train, y_valid, y_test, min_max_transformer = min_max_scaler(y_train, y_valid, y_test)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer

def standard_preprocessor(X, y, train_size=0.8, valid_size=0.1):
	"""
	Preprocess the data so it is ready for machine learning

	Preprocessing steps

		#. Remove extra dimensions from the matrix of predictors if the
		simulation is 2D
		#. It splits the data set into training, validation and test data sets
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
	X = remove_extra_dimesions(X)
	(X_train, X_valid, X_test,
	y_train, y_valid, y_test) = train_valid_test_split(X, y, train_size, valid_size)
	X_train, X_valid, X_test = full_standard_scaler(X_train, X_valid, X_test)
	y_train, y_valid, y_test, min_max_transformer = min_max_scaler(y_train, y_valid, y_test)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer

def preprocessor(X, y, preprocessor_type="physical", train_size=0.8, valid_size=0.1):
	"""
	Preprocess the data so it is ready for machine learning

	Parameters
	----------
	X : numpy.ndarray
		5D array representing the matrix of predictors
	y : numpy.ndarray
		1D array representing the vector of responses
	preprocessor_type : str, default="physical"
		The type of preprocessor to use. Options are "physical","standard","dumb".
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
	if preprocessor_type == "physical":
		function = physical_preprocessor
	elif preprocessor_type == "standard":
		function = standard_preprocessor
	elif preprocessor_type == "dumb":
		function = dumb_preprocessor
	X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer = function(X,
		y.reshape(-1,1),
		train_size,
		valid_size)
	return X_train, X_valid, X_test, y_train, y_valid, y_test, min_max_transformer










