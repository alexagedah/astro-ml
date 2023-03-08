"""
The read module provides functions for reading data from HDF5 files into 
numpy.ndarrays.
"""
import os
import pathlib

import h5py
import numpy as np

def get_hdf5_files(data_relative_path):
	"""
	Return a list of the HDF5 files in the folder containing simulation data

	Parameters
	----------
	path : pathlib.Path
		A relative path to the folder containing the simulation data

	Returns
	-------
	hdf5_files : list of str
		Sorted list of the name of HDF5 files in the folder
	"""
	data_absolute_path = pathlib.Path.cwd() / data_relative_path
	files = os.listdir(data_absolute_path)
	hdf5_files = []
	for file in files:
		if file.endswith(".h5"):
			hdf5_files.append(file)
	hdf5_files.sort()
	return hdf5_files

def get_time_coordinates(data_relative_path):
	"""
	Returns a dictionary with the times for a simulation

	This function assumes that time = 0 corresponds to the file that appears
	first alphabetically.

	Parameters
	----------
	data_relative_path : pathlib.Path
		A relative path to the folder containing the simulation data

	Returns
	-------
	t : numpy.ndarray
		1D numpy.ndarray representing the time coordinates
	"""
	hdf5_files = get_hdf5_files(data_relative_path)
	t = np.array(range(len(hdf5_files)))
	return t

def get_cell_coordinates(data_relative_path):
	"""
	Return a dictionary with the coordinates of the cells in the simulation

	This function returns a dictionary with the coordinates of the all the cells
	in a simulation. The keys specify the axis the coordinates are for and the
	values specifies the coordinates

		* The 0th axis represents the x-direction
		* The 1st axis represents the y-direction
		* The 2nd axis represents the z-direction

	This function gets the coordinates from a single output file
	(and we assume) that all files have the same coordinates.

	Parameters
	----------
	data_relative_path : pathlib.Path
		A relative path to the folder containing the simulation data

	Returns
	-------
	x : numpy.ndarray
		3D numpy.ndarray representing the x-coordiantes
	y : numpy.ndarray
		3D numpy.ndarray representing the y-coordiantes
	z : numpy.ndarray
		3D numpy.ndarray representing the z-coordiantes
	"""
	file_path = data_relative_path / pathlib.Path("data.0000.flt.h5")
	file_object = h5py.File(file_path, "r")
	cell_coords_group = file_object["cell_coords"]
	x = cell_coords_group["X"][:]
	y = cell_coords_group["Y"][:]
	z = cell_coords_group["Z"][:]
	if x.ndim == 2:
		return x[:,:,np.newaxis], y[:,:,np.newaxis], z[:,:,np.newaxis]
	else:
		return x, y, z

def get_variables_at_snapshot(file_object):
	"""
	Return the values of the variables in each of the cells at a moment in time
	in a PLUTO simulation

	Parameters
	----------
	file_object : h5py.File
		File object representing a hdf5 file with simulation data at a 

	Returns
	-------
	B_x_t : numpy.ndarray
		3D array containing the x component of the magnetic field at each cell
	B_y_t : numpy.ndarray
		3D array containing the y component of the magnetic field at each cell
	B_z_t : numpy.ndarray
		3D array containing the z component of the magnetic field at each cell
	p_t : numpy.ndarray
		3D array containing the pressure at each cell
	rho_t : numpy.ndarray
		3D array containing the density at each cell
	v_x_t : numpy.ndarray
		3D array containing the x component of the velocity at each cell
	v_y_t : numpy.ndarray
		3D array containing the y component of the velocity at each cell
	v_z_t : numpy.ndarray
		3D array containing the y component of the velocity at each cell
	"""
	timestep_key = list(file_object.keys())[0]
	timestep_group = file_object[timestep_key]
	vars_group = timestep_group["vars"]
	B_x_t = vars_group["Bx1"][:][:,:,np.newaxis]
	B_y_t = vars_group["Bx2"][:][:,:,np.newaxis]
	B_z_t = vars_group["Bx3"][:][:,:,np.newaxis]
	p_t = vars_group["prs"][:][:,:,np.newaxis]
	rho_t = vars_group["rho"][:][:,:,np.newaxis]
	v_x_t = vars_group["vx1"][:][:,:,np.newaxis]
	v_y_t = vars_group["vx2"][:][:,:,np.newaxis]
	v_z_t = vars_group["vx3"][:][:,:,np.newaxis]
	return B_x_t, B_y_t, B_z_t, p_t, rho_t, v_x_t, v_y_t, v_z_t

def get_fluid_variables(data_relative_path):
	"""
	Return a dictionary containing all the variables which describe the fluid

	This function returns a dictionary with all the variables which describe the
	fluid. The data in the ideal format for exploration. The keys of the
	dictionary specify the fluid variable.
	The values are 4D numpy.ndarrays where

		* The 0th axis specifies the x-coordinate
		* The 1st axis specifies the y-coordinate
		* The 2nd axis specifies the z-coordinate
		* The 3rd axis specifies the time

	Parameters
	----------
	data_relative_path : pathlib.Path
		A relative path to the folder containing the simulation data

	Returns
	-------
	fluid_variables  : dict of {str: numpy.ndarray}
		Dictionary where each string specifies the variable and each value is
		a 4D numpy.ndarray representing the variable. The keys in the dictionary
		are B_x, B_y, B_z, p, rho, v_x, v_y and v_z

	"""
	hdf5_files = get_hdf5_files(data_relative_path)

	B_x_list = []
	B_y_list = []
	B_z_list = []
	p_list = []
	rho_list = []
	v_x_list = []
	v_y_list = []
	v_z_list = []

	for file_name in hdf5_files:
		file_path = data_relative_path / pathlib.Path(file_name)
		file_object = h5py.File(file_path, "r")
		(B_x_t, B_y_t, B_z_t, p_t,
		rho_t, v_x_t, v_y_t, v_z_t) = get_variables_at_snapshot(file_object)
		B_x_list.append(B_x_t[:,:,:,np.newaxis])
		B_y_list.append(B_y_t[:,:,:,np.newaxis])
		B_z_list.append(B_z_t[:,:,:,np.newaxis])
		p_list.append(p_t[:,:,:,np.newaxis])
		rho_list.append(rho_t[:,:,:,np.newaxis])
		v_x_list.append(v_x_t[:,:,:,np.newaxis])
		v_y_list.append(v_y_t[:,:,:,np.newaxis])
		v_z_list.append(v_z_t[:,:,:,np.newaxis])

	B_x = np.concatenate(B_x_list, axis = 3)
	B_y = np.concatenate(B_y_list, axis = 3)
	B_z = np.concatenate(B_z_list, axis = 3)
	p = np.concatenate(p_list, axis = 3)
	rho = np.concatenate(rho_list, axis = 3)
	v_x = np.concatenate(v_x_list, axis = 3)
	v_y = np.concatenate(v_y_list, axis = 3)
	v_z = np.concatenate(v_z_list, axis = 3)
	fluid_variables = {
	"B_x":B_x,
	"B_y":B_y,
	"B_z":B_z,
	"p":p,
	"rho":rho,
	"v_x":v_x,
	"v_y":v_y,
	"v_z":v_z
	}
	return fluid_variables



