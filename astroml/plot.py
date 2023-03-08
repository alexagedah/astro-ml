"""
The plot module contains functions for producing graphs
"""
# Standard Library
import pathlib
# 3rd Party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "Times New Roman"
plt.style.use("default")

def save_show_figure(figure, save_figure, filename=None, show_figure=False):
	"""
	Save and/or show a figure

	Parameters
	----------
	figure : matplotlib.figure
		The figure to save or show
	save_figure : bool, default=False
		Whether to save the figure
	filename : str, default=None
		The name of the file to save the figure to. The default is None for
		cases where you don't want to save the figure
	show_figure : bool, default=False
		Whether to show the figure
	"""
	if save_figure:
		figure.savefig(filename)
	if show_figure:
		plt.show()
	plt.close()

def plot_single_distribution(data_set, variable, variable_name, time=None, save_fig=True, show_fig=False):
	"""
	Plot a histogram showing the distribution of a variable at a specific moment 
	in time

	Parameters
	----------
	data_set : str
		A relative path to the data set to produce histograms for
	variable : numpy.ndarray
		4D numpy.ndarray represnting the variable
	variable_name : numpy.ndarray
		The name of the variable
	time : int, default=None
		The time to plot the histogram for. The default None plots for the
		variable across all time
	save_fig : bool, default=True
		Whether to save the figure
	show_fig : bool, default=False
		Whether to show the figure
	"""
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel(variable_name)
	ax.set_ylabel("Frequency")
	if time == None:
		ax.set_title(f"The Distribution of {variable_name}")
		ax.hist(variable.flatten(), bins = 50)
	else:
		ax.set_title(f"The Distribution of the {variable_name} at time = {time}")
		ax.hist(variable[:,:,:,time].flatten(), bins = 50)
	filename = f"exploration/{data_set}/{variable_name}_single_distribution"
	save_show_figure(fig, save_fig, filename, show_fig)

def plot_distributions(data_set, variable, variable_name, save_fig=True, show_fig=False):
	"""
	Plot the distribution of a variable at various times

	Parameters
	----------
	data_set : str
		A relative path to the data set to produce histograms for
	variable : numpy.ndarray
		4D numpy.ndarray representing the variable in the simulation
	variable_name : numpy.ndarray
		The name of the variabele
	save_fig : bool, default=True
		Whether to save the figure
	show_fig : bool, default=False
		Whether to show the figure
	"""
	times = np.linspace(0, variable.shape[-1]-1, 9).astype(np.int64)
	array = variable.reshape(-1, variable.shape[-1])[:,times]
	mpl.rc('xtick', labelsize=4) 
	mpl.rc('ytick', labelsize=4) 
	fig, axes = plt.subplots(3,3, figsize=(16,10))
	for i in range(3):
		for j in range(3):
			time = times[i*3+j]
			axes[i,j].hist(array[:,i*3+j], bins = 100)
			axes[i,j].set_title(f"The Distribution of {variable_name} at time = {time}", {'fontsize':10})
			axes[i,j].set_xlabel("B_x", {'fontsize': 4})
			axes[i,j].set_ylabel("Frequency", {'fontsize': 4})
	filename = f"exploration/{data_set}/{variable_name}_distributions"
	save_show_figure(fig, save_fig, filename, show_fig)

def plot_single_contour(data_set, variable, variable_name, X, Y, z=0, time=None, save_fig=True, show_fig=False):
	"""
	Produce a contour plot for a variable at a specific time

	Parameters
	----------
	data_set : str
		A relative path to the data set to produce histograms for
	variable : numpy.ndarray
		4D numpy.ndarray represnting the variable
	variable_name : numpy.ndarray
		The name of the variable
	z : int, default=0
		The z-coordinate to plot the variable at. The default is 0 which plots
		the variable in the z = 0 plane. This should be used if the data set is
		2D.
	time : int, default=None
		The time to plot the histogram for. The default None plots for the
		variable across all time
	save_fig : bool, default=True
		Whether to save the figure
	show_fig : bool, default=False
		Whether to show the figure
	"""
	Z = variable[:,:,z,time]
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_title(f"{variable_name} at Time t = {time}")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	CS = ax.contour(X[:], Y[:],Z[:]) 
	ax.clabel(CS, inline=True, fontsize=5)
	ax.grid(True)
	filename = f"exploration/{data_set}/{variable_name}_distribution"
	save_show_figure(fig, save_fig, filename, show_fig)

def plot_contours(data_set, variables, variable_name, z=0, save_fig=True, show_fig=False):
	"""
	Create contour plots for the variable at various times

	Parameters
	----------
	data_set : str
		A relative path to the data set to produce histograms for
	variables : dict of {str:numpy.ndarray, str: dict of {str:numpy.ndarray} }
		4D numpy.ndarray represnting the variable
	variable_name : numpy.ndarray
		The name of the variable to plot contours for
	z : int, default=0
		The z-coordinate to plot the variable at. The default is 0 which plots
		the variable in the z = 0 plane. This should be used if the data set is
		2D.
	time : int, default=None
		The time to plot the histogram for. The default None plots for the
		variable across all time
	save_fig : bool, default=True
		Whether to save the figure
	show_fig : bool, default=False
		Whether to show the figure
	"""
	X = variables["X"]
	Y = variables["Y"]
	variable = variables["Fluid Variables"][variable_name]

	times = np.linspace(0, variable.shape[-1]-1, 9).astype(np.int64)
	mpl.rc('xtick', labelsize=4) 
	mpl.rc('ytick', labelsize=4) 
	fig, axes = plt.subplots(3,3, figsize=(16,10))
	for i in range(3):
		for j in range(3):
			time = times[i*3+j]
			axes[i,j].set_title(f"{variable_name} at Time t = {time}")
			axes[i,j].set_xlabel("X", {'fontsize': 4})
			axes[i,j].set_ylabel("Y", {'fontsize': 4})
			CS = axes[i,j].contour(X[:,:,z], Y[:,:,z],variable[:,:,z,time]) 
			axes[i,j].clabel(CS, inline=True, fontsize=5)
			axes[i,j].grid(True)
	filename = f"exploration/{data_set}/{variable_name}_contours"
	save_show_figure(fig, save_fig, filename, show_fig)

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



