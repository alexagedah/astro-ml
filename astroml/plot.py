"""
The plot module contains functions for producing graphs
"""
# Standard Library
import pathlib
# 3rd Party
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["font.family"] = "Times New Roman"
plt.style.use("default")

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
