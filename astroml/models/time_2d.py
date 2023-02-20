"""
Module for building artificial neural networks to predict the time a snapshot
from the simulation came from
"""
# Import libraries
from functools import partial

import numpy as np
import tensorflow as tf
import keras_tuner as kt
tf.random.set_seed(42)

def regression_mlp(X_train):
	"""
	Build a regression multilayer perceptron to predict the time of a snapshot

	Parameters
	----------
	X_train : numpy.ndarray
		Design matrix
	Returns
	-------
	model : tf.keras.Model
	model_name : str
		The name of the model. This determines what folder the checkpoints and
		final model is saved to.
	"""
	grid_size = X_train.shape[1]
	model_name = "regression_mlp"
	model = tf.keras.Sequential([
		tf.keras.layers.Flatten(input_shape=X_train.shape[1:]),
		tf.keras.layers.Dense(20, activation="relu"),
		tf.keras.layers.Dense(20, activation="relu"),
		tf.keras.layers.Dense(20, activation="relu"),
		tf.keras.layers.Dense(20, activation="relu"),
		tf.keras.layers.Dense(20, activation="relu"),
		tf.keras.layers.Dense(20, activation="relu"),
		tf.keras.layers.Dense(1)
		])
	model.compile(loss="mse",
	optimizer="adam")
	return model, model_name

def hp_regression_mlp(hp):
	"""
	Build a regression multilayer perceptron

	Parameters
	----------
	hp : keras_tuner.HyperParameters

	Returns
	-------
	model : tf.keras.Model
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

def regression_cnn(X_train):
	"""
	Build a regression convolutional neurlal network

	X_train : numpy.ndarray
		Training design matrix
	Returns
	-------
	model : tf.keras.Model
	model_name : str
	"""
	model_name = "time_2d_regression_cnn"
	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(64, 3, padding="same",activation="relu", input_shape=X_train.shape[1:]),
		tf.keras.layers.Conv2D(64, 3, padding="same",activation="relu"),
		tf.keras.layers.MaxPool2D(),
		tf.keras.layers.Conv2D(128, 3, padding="same",activation="relu"),
		tf.keras.layers.Conv2D(128, 3, padding="same",activation="relu"),
		tf.keras.layers.MaxPool2D(),
		tf.keras.layers.Dense(64, activation="relu"),
		tf.keras.layers.Dense(1)
	])
	model.compile(loss="mse", optimizer="adam")
	return model, model_name

def hp_regression_cnn(hp):
	"""
	Build a regression convolutional neural network with hyperparameters that can
	be tuned
	 
	Parameters
	----------
	hp : keras_tuner.HyperParameters

	Returns
	-------
	model : tf.keras.Model
	"""
	n_hidden = hp.Int("n_hidden", min_value=1, max_value=8, step=1)
	filter_size = hp.Int("filter_size", min_value=3, max_value=5, step=2)
	model = tf.keras.Sequential()
	for i in range(n_hidden, 0, -1):
		model.add(tf.keras.layers.Conv2D(512/(2*i), filter_size, padding="same",activation="relu"))
		model.add(tf.keras.layers.Conv2D(512/(2*i), filter_size, padding="same",activation="relu"))
		model.add(tf.keras.layers.MaxPool2D())
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(64, activation="relu")),
	model.add(tf.keras.layers.Dense(64, activation="relu")),
	model.add(tf.keras.layers.Dense(1))
	model.compile(loss="mse", optimizer="adam")
	return model






