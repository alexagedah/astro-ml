"""
Module containing functions for training artificial neural networks
"""
import pathlib
import os

import tensorflow as tf

from . import plot

def train_model(model, model_name, X_train, X_valid, y_train, y_valid):
	"""
	Train the model using checkpoints during training and early stopping and save
	the learning curve

	Parameters
	----------
	model : tf.keras.Model
	model_name : str
	X_train : numpy.ndarray
	X_valid : numpy.ndarray
	y_train : numpy.ndarray
	y_valid : numpy.ndarray
	"""
	checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(f"astroml/training/{model_name}/my_checkpoints",
                                                  save_weights_only = True)
	early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
	                                                    restore_best_weights=True)

	history = history = model.fit(X_train, y_train, epochs = 1000,
								validation_data=(X_valid, y_valid),
								callbacks = [checkpoint_cb, early_stopping_cb])
	model.save(f"astroml/saved_models/{model_name}")
	plot.plot_learning_curve(history, model_name)