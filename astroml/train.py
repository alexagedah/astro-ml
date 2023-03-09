"""
The train model contains functions for training artificial neural networks
"""
# Standard Libary
import os
import pathlib
import shutil
# 3rd Party
import tensorflow as tf
# Libary Specific
from . import plot

tf.random.set_seed(42)

def train_model(model, model_name, X_train, X_valid, y_train, y_valid, epochs=1000):
	"""
	Train the model, save it and plot the learning curve for the model

	This function is ideal for training custom models.

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
	model.save(f"saved_models/{model_name}")
	clear_folder("training")
	plot.plot_learning_curve(history, model_name)

def clear_folder(folder):
	"""
	Clear everything in a folder

	Parameters
	----------
	folder : str
		The name of the folder to clear
	"""
	for filename in os.listdir(folder):
	    file_path = os.path.join(folder, filename)
	    try:
	        if os.path.isfile(file_path) or os.path.islink(file_path):
	            os.unlink(file_path)
	        elif os.path.isdir(file_path):
	            shutil.rmtree(file_path)
	    except Exception as e:
	        print('Failed to delete %s. Reason: %s' % (file_path, e))

