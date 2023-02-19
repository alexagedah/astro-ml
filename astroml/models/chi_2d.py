"""
Module for building artifical neural networks to predict the magnetisation
of the disc
"""
import tensorflow as tf

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
	model_name = "chi_2d_regression_mlp"
	model = tf.keras.Sequential([
		tf.keras.layers.Flatten(input_shape=(grid_size, grid_size, 6)),
		tf.keras.layers.Dense(20, activation="relu"),
		tf.keras.layers.Dense(20, activation="relu"),
		tf.keras.layers.Dense(20, activation="relu"),
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