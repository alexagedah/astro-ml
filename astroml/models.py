"""
The models module contains functions for building artificial neural networks.
"""
import keras_tuner as kt
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)

def get_regression_mlp(input_shape, preprocessor_type, feature_list, response, n_hidden_layers=4, n_hidden_units=256):
	"""
	Return a multilayer perceptron for regression

	Parameters
	----------
	input_shape : int
		The shape of the inputs
	preprocessor_type : str
		The type of preprocessor to use. Options are "simple","standard","smart".
	feature_list : list of str
		A list of the additional features that have been used
	response : str
		The name of the response variable 
	n_hidden_layers : int, default=4
		The number of hidden layers
	n_hidden_units : int, default=256
		The number of hidden units

	Returns
	-------
	model : tensorflow.keras.Model
		The multilayer perceptron for regression
	model_name : str
		The name of the model. This determines what folder the checkpoints and
		final model is saved to. The format for the name of the model is 
		<grid_size>_<n_hidden_layers>_<n_hidden_units>_<response>

	"""
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Input(input_shape))
	model.add(tf.keras.layers.Flatten())
	for _ in range(n_hidden_layers):
		model.add(tf.keras.layers.Dense(n_hidden_units, activation="relu"))
	model.add(tf.keras.layers.Dense(1))
	model.compile(loss="mse",
	optimizer="adam")
	features = '-'.join(feature_list)
	model_name = f"{input_shape[0]}_{preprocessor_type}_{features}_{n_hidden_layers}_{n_hidden_units}_{response}"
	return model, model_name

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
