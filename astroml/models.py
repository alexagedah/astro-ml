"""
The models module contains functions for building artificial neural networks.
"""
import keras_tuner as kt
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)

def get_regression_mlp(input_shape, n_hidden_layers, n_hidden_units):
	"""
	Return a multilayer perceptron for regression

	Parameters
	----------
	input_shape : int
		The shape of the inputs
	n_hidden_layers : int
		The number of hidden layers
	n_hidden_units : int
		The number of hidden units

	Returns
	-------
	model : tensorflow.keras.Model
		The multilayer perceptron for regression
	"""
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Input(input_shape))
	model.add(tf.keras.layers.Flatten())
	for _ in range(n_hidden_layers):
		model.add(tf.keras.layers.Dense(n_hidden_units, activation="relu"))
	model.add(tf.keras.layers.Dense(1))
	model.compile(loss="mse",
	optimizer="adam")
	return model

def get_model_name(data_set_name, observation_size, features, response, feature_scaling, response_scaling, n_hidden_layers, n_hidden_units):
	"""
	Return the name of a model

	Parameters
	----------
	data_set_name : str
		The name of the data set
	observation_size : int
		The length of each 2D grid or 3D cube
	features : list of str
		The list of features to use
	response : str
		The name of the response variable 
	feature_scaling : default
		The scaling that should be applied to the features
	response_scaling : default
		The scaling that should be applied to the responses
	n_hidden_layers : int
		The number of hidden layers
	n_hidden_units : int
		The number of hidden units

	Returns
	-------
	model_name : str
		The name of the model.
		<data_set>_<observation_size>_{features}_<response>_<feature_scaling>_<response_scaling>_<n_hidden_layers>_<n_hidden_units>
	"""
	names_components = [data_set_name,
	str(observation_size),
	'-'.join(features),
	response,
	feature_scaling,
	response_scaling,
	str(n_hidden_layers),
	str(n_hidden_units)]
	model_name = '_'.join(names_components)
	return model_name

def get_hp_regression_mlp(hp):
	"""
	Return a multilayer perceptron for regression with hyperparameters defined

	Parameters
	----------
	hp : keras_tuner.HyperParameters

	Returns
	-------
	model : tensorflow.keras.Model
	"""
	n_hidden = hp.Int("n_hidden", min_value=4, max_value=16, step=2, sampling="log")
	n_neurons = hp.Int("n_neurons", min_value=32, max_value=256, step=2, sampling="log")
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Flatten())
	for _ in range(n_hidden):
	    model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
	model.add(tf.keras.layers.Dense(1))
	model.compile(loss="mse", optimizer="adam")
	return model
