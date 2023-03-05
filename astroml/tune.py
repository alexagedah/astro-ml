"""
Module for tuning hyperparameters for models.
"""
import tensorflow as tf
import keras_tuner as kt


tf.random.set_seed(42)

def tune_model(build_model, model_name, X_train, X_valid, y_train, y_valid, overwrite=False, max_trials=10):
	"""
	Perform hyperparameter tuning for a model and display the results

	Parameters
	----------
	build_model : function
	model_name : str
	X_train : numpy.ndarray
	X_valid : numpy.ndarray
	y_train : numpy.ndarray
	y_valid : numpy.ndarray
	overwrite : bool, default=False
		If False, reloads an existing project of the same name if one is found. 
		Otherwise overwrites the project
	max_trials : int, default=10
		The total number of trials (model configurations) to test at most
	"""
	early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
	                                                    restore_best_weights=True)
	random_search_tuner = kt.RandomSearch(
								build_model,
								objective="val_loss",
								max_trials = max_trials,
								overwrite=overwrite,
								directory="hp_tuning",
								project_name=model_name,
								)
	random_search_tuner.search(X_train, y_train, epochs =1000,
		validation_data=(X_valid, y_valid),
		callbacks = [early_stopping_cb])
	random_search_tuner.search_space_summary(extended=True)
	print()
	random_search_tuner.results_summary()

def load_tuning_results(build_model, model_name):
	"""
	Display the results from hyperparameter tuning for a model

	Parameters
	----------
	build_model : function
	model_name : str
	"""
	random_search_tuner = kt.RandomSearch(
								build_model,
								objective="val_loss",
								overwrite=False,
								directory="hp_tuning",
								project_name=model_name,
								)
	random_search_tuner.search_space_summary(extended=True)
	print()
	random_search_tuner.results_summary()


