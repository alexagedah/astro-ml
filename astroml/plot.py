"""
The plot module contains functions for producing graphs
"""
# Standard Library
import pathlib
# 3rd Party
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

mpl.rcParams["font.family"] = "Times New Roman"
plt.style.use("default")

def plot_distribution(variable, variable_name, time=None):
	"""
	Plot a histogram showing the distribution of a variable at a specific
	moment in time

	Parameters
	----------
	variable : numpy.ndarray
		4D numpy.ndarray represnting the variable
	variable_name : numpy.ndarray
		The name of the variable
	time : int, default=None
		The time to plot the histogram for. The default is None plots for the
		variable across all time
	"""
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel(variable_name)
	ax.set_ylabel("Frequency")
	if time == None:
		ax.set_title(f"Distribution of the {variable_name} of the Disc For All Times")
		ax.hist(variable.flatten(), bins = 50)
	else:
		ax.set_title(f"Distribution of the {variable_name} of the Disc at Time t = {time}")
		ax.hist(variable[:,:,:,time].flatten(), bins = 50)
	plt.show()

def contour_plot(variable, variable_name, time, z=0):
	"""
	Produce a contour plot for a variable at a specific time

	Parameters
	----------
	variable : numpy.ndarray
		4D numpy.ndarray representing the variable
	variable_name : numpy.ndarray
		The name of the variable
	time : int
		The time to plot the variable for
	z : int
		The z-coordinate to plot the variable at. The default is 0 which plots
		the variable in the z = 0 plane. This should be used if the data set is
		2D.
	"""
	Z = variable[:,:,0,time]
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_title(f"{variable_name} of the Disc at Time t = {time}")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	CS = ax.contour(X[:], Y[:],Z[:]) 
	ax.clabel(CS, inline=True, fontsize=5)
	ax.grid(True)
	plt.show()

def plot_learning_curve(history, model_name, show=False):
	"""
	Plot the learning curve for a artificial neural network during training and
	save the file to the learning_curves folder

	Parameters
	----------
	history : tf.keras.callbacks.History
	model_name : str
		The name of the model
	show : bool, default=False
		Whether the learning curve should be displayed
	"""
	history_df = pd.DataFrame(history.history)
	fig = plt.figure(figsize=(10,6))
	ax1 = fig.add_subplot(1,1,1)
	ax1.set_title("Learning Curve")
	ax1.set_ylabel("Loss")
	ax1.set_xlabel("Epoch")
	ax1.plot(history_df.loc[:,"loss"], label="Training Loss")
	ax1.plot(history_df.loc[:,"val_loss"], label="Validation Loss")
	ax1.set_xlim(0, len(history_df))
	ax1.set_ylim(0, history_df.values.flatten().max())
	ax1.legend()
	ax1.grid(True)
	if show:
		plt.show()
	save_path = pathlib.Path("learning_curves") / pathlib.Path(model_name)
	fig.savefig(save_path)
