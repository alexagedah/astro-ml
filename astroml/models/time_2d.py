"""
Module for building artificial neural networks to predict the time a snapshot
from the simulation came from
"""
# Import libraries
from functools import partial

import tensorflow as tf
tf.random.set_seed(42)

def regression_mlp(grid_size):
	"""
	Build a feed forward neural netwrok to predict the time of a snapshot

	Parameters
	----------
	grid_size : int

	Returns
	-------
	model : tf.keras.Model
	model_name : str
		The name of the model. This determines what folder the checkpoints and
		final model is saved to.
	"""
	model_name = "time_2d_regression_mlp"
	model = tf.keras.Sequential([
		tf.keras.layers.Normalization(),
		tf.keras.layers.Flatten(input_shape=(grid_size, grid_size)),
		tf.keras.layers.Dense(50, activation="relu"),
		tf.keras.layers.Dense(50, activation="relu"),
		tf.keras.layers.Dense(1)
		])
	model.compile(loss="mse",
	optimizer="adam")
	return model, model_name

def regression_cnn(grid_size):
	"""
	Parameters
	----------
	grid_size : int

	Build a CNN to predict the time of a snapshot
	"""
	model_name = "time_2d_regression_cnn"
	DefaultConv2D = partial(tf.keras.layers.Conv2D,
			kernel_size = 3,
			padding = "same",
			activation="relu",
			kernel_initializer="he_normal")
	model = tf.keras.Sequential([
		tf.keras.layers.Normalization(),
		DefaultConv2D(filters=64, kernel_size = 3, input_shape=(grid_size, grid_size)),
		tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
		DefaultConv2D(filters = 128),
		DefaultConv2D(filters = 128),
		tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation = "relu",
			kernel_initializer="he_normal"),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(64, activation = "relu",
			kernel_initializer="he_normal"),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(1)
		])
	model.compile(loss="mse",
	optimizer="adam")
	return model

def classification_cnn(grid_size, number_of_classes):
	"""
	Create a CNN to predict the time of a snapshot

	Parameters
	----------
	grid_size : int
	number_of_classes : int
	"""
	model_name = "time_2d_classification_cnn"
	DefaultConv2D = partial(tf.keras.layers.Conv2D,
			kernel_size = 3,
			padding = "same",
			activation="relu",
			kernel_initializer="he_normal")

	model = tf.keras.Sequential([
		tf.keras.layers.Normalization(),
		DefaultConv2D(filters=64, kernel_size = 3, input_shape=(grid_size,grid_size)),
		tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
		DefaultConv2D(filters = 128),
		DefaultConv2D(filters = 128),
		tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation = "relu",
			kernel_initializer="he_normal"),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(64, activation = "relu",
			kernel_initializer="he_normal"),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(len(np.unique(y_train)), activation = "softmax")
		])
	model.compile(loss="sparse_categorical_crossentropy",
	optimizer="adam",
	metrics=["accuracy"])
	return model





