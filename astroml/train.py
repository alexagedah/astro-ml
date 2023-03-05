"""
The train model contains functions for training artificial neural networks
"""
# Standard Libary
import pathlib
import os
# 3rd Party
import tensorflow as tf
# Libary Specific
from . import plot

tf.random.set_seed(42)

def train_model(model, model_name, X_train, X_valid, y_train, y_valid, norm_layer, epochs=1000):
	"""
	Train the model, save it and plot the learning curve for the model

	Parameters
	----------
	model : tf.keras.Model
		The model to train
	model_name : str
		The name of the model.
	X_train : numpy.ndarray
		numpy.ndarray representing the matrix of predictors for the training
		data set
	X_valid : numpy.ndarray
		numpy.ndarray representing the matrix of predictors for the validation
		data set
	y_train : numpy.ndarray
		numpy.ndarray representing the vector of responses for the training
		data set
	y_valid : numpy.ndarray
		numpy.ndarray representing the vector of responses for the validation
		data set
	norm_layer : tf.keras.layers.Normalization
		The fitted normalisation layer
	epochs: int, default=1000
		The number of epochs to train for
	"""
	checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f"training/{model_name}/my_checkpoints",
                                                  save_weights_only = True)
	early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
	                                                    restore_best_weights=True)
	history = model.fit(X_train, y_train, epochs=epochs,
								validation_data=(X_valid, y_valid),
								callbacks = [checkpoint_cb, early_stopping_cb])
	model.save(f"astroml/saved_models/{model_name}")
	plot.plot_learning_curve(history, model_name)