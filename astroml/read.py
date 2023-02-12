"""
Module for reading data from HDF5 files into 4D NumPy arrays
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
	Returns the times in the simulation

	Parameters
	----------
	data_relative_path : pathlib.Path
		A relative path to the folder containing the simulation data

	Returns
	-------
	T : numpy.ndarray
		1D array containing all the time coordinates in the simulation

	TODO:
	Should I actually get the times from the file names rather than assuming
	the times start from 0?
	"""
	hdf5_files = get_hdf5_files(data_relative_path)
	T = np.array(range(len(hdf5_files)))
	return T

def get_cell_coordinates(data_relative_path):
	"""
	Return the coordinates of the cells in a PLUTO simulation

	This function returns the coordinates of the cells in a PLUTO simulation.
	This function gets the coordinates from a single output file (and we assume)
	that all files have the same coordinates 

	Parameters
	----------
	data_relative_path : pathlib.Path
		A relative path to the folder containing the simulation data

	Returns
	-------
	X : numpy.ndarray
		3D array containing the x coordinates of each point in simulation
	Y : numpy.ndarray
		3D array containing the y coordinates of each point in the simulation
	Z : numpy.ndarray
		3D array containing the z coordinates of each point in the simulation
	"""
	file_path = data_relative_path / pathlib.Path("data.0000.flt.h5")
	file_object = h5py.File(file_path, "r")
	cell_coords_group = file_object["cell_coords"]
	X = cell_coords_group["X"][:]
	Y = cell_coords_group["Y"][:]
	Z = cell_coords_group["Z"][:]
	if X.ndim == 2:
		return X[:,:,np.newaxis], Y[:,:,np.newaxis], Z[:,:,np.newaxis]
	else:
		return X, Y, Z

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

def get_variables(data_relative_path):
	"""
	Return the variables in a PLUTO simulation

	This function returns a list of numpy.ndarrays which each contain all the
	variables in a PLUTO simulation. The data from the PlUTO simulation should
	be contained in a folder called data in the current working directory.
	For each array that is returned:
		The 0th axis represents the x-direction
		The 1st axis represents the y-direction
		The 2nd axis represents the z-direction
		The 3rd axis represents the time

	Parameters
	----------
	data_relative_path : pathlib.Path
		A relative path to the folder containing the simulation data

	Returns
	-------
	B_x : numpy.ndarray
		4D array with x component of the magnetic field at each cell
	B_y : numpy.ndarray
		4D array with the y component of the magnetic field at each cell
	B_z : numpy.ndarray
		4D array with the z component of the magnetic field at each cell
	p : numpy.ndarray
		4D array with gas pressure at each cell
	rho : numpy.ndarray
		4D array with denstiy at each cell
	v_x : numpy.ndarray
		4D array with x component of the velocity at each cell
	v_y : numpy.ndarray
		4D array with y component of the velocity at each cell
	v_z : numpy.ndarray
		4D array with z component of the velocity at each cell
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
	return B_x, B_y, B_z, p, rho, v_x, v_y, v_z

def create_snapshot_array(B_x_t, B_y_t, B_z_t, p_t, rho_t, v_x_t, v_y_t, v_z_t):
	"""
	Create a 4D NumPy array containing information on all the variables at a
	certain moment in time

	Parameters
	----------
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

	Returns
	-------
	snapshot_array : numpy.ndarray
		4D array containing all the variables at a certain moment in time
		The 0th axis represents the x-direction
		The 1st axis represents the y-direction
		The 2nd axis represents the z-direction
		The 3rd axis specifies the 8 different variables
	"""
	snapshot_array = np.concatenate([B_x_t[:,:,:,np.newaxis],
		B_y_t[:,:,:,np.newaxis],
		B_z_t[:,:,:,np.newaxis],
		p_t[:,:,:,np.newaxis],
		rho_t[:,:,:,np.newaxis],
		v_x_t[:,:,:,np.newaxis],
		v_y_t[:,:,:,np.newaxis],
		v_z_t[:,:,:,np.newaxis]], axis = 3)
	return snapshot_array

def split_snapshot_array(snapshot_array, grid_size = 5, periodic_bc = False):
	"""
	Create a NumPy array containing observations of different sections of a
	simulation

	Parameters
	----------
	snapshot_array : numpy.ndarray
		4D array containing all the variables at a certain moment in time
	grid_size : int
		The size of each grid to consider as an observation
	Returns
	-------
	X_t : numpy.ndarray
		5D array containing observations of parts of the disc at a certain moment
	int time
			The 0th axis specifies the observation
			The 1st axis represents the x-direction
			The 2nd axis represents the y-direction
			The 3rd axis represents the z-direction
			The 4th axis represents the 8 different predictor variables

	TODO: UNIT TESTING
		There should be certain relationships between different observations!
		Have tests to make sure the array has been formed correctly
	"""
	X_t_list = []
	x_cells = snapshot_array.shape[0]
	y_cells = snapshot_array.shape[1]
	z_cells = snapshot_array.shape[2]
	for x_start in range(0, x_cells):
		x_end = x_start + grid_size
		if x_end == x_cells:
			continue
		elif x_end > x_cells:
			continue

		for y_start in range(0, y_cells):
			y_end = y_start + grid_size
			if y_end == y_cells:
				continue
			elif y_end > y_cells:
				continue

			for z_start in range(0, z_cells):
				if z_cells == 1:
					observation = np.expand_dims(
						snapshot_array[x_start:x_end,y_start:y_end,:,:],
					0
					)
					X_t_list.append(observation)
	X_t = np.concatenate(X_t_list, axis = 0)
	return X_t

def get_observations(data_relative_path, grid_size=5, response="Time"):
	"""
	Return a design matrix and a response vector

	Parameters
	----------
	data_relative_path : pathlib.Path
		A relative path to the folder containing the simulation data
	grid_size : int
		The size of each grid to consider as an observation
	response : str, default="Time"
		The response variable that you are trying to predict. Options are "Time"
		and "Magnetic Field Strength" (TODO)

	Returns
	-------
	X : numpy.ndarray
		5D array representing the design matrix
			The 0th axis specifies the observation
			The 1st axis represents the x-direction
			The 2nd axis represents the y-direction
			The 3rd axis represents the z-direction
			The 4th axis represents the 8 different predictor variables
	y : numpy.ndarray
		1D array representing the response vector
	"""
	x_observation_list = []
	y_observation_list = []

	hdf5_files = get_hdf5_files(data_relative_path)
	for file_name in hdf5_files:
		t = int(file_name[5:9])
		file_path = data_relative_path / pathlib.Path(file_name)
		file_object = h5py.File(file_path, "r")
		(B_x_t, B_y_t, B_z_t, p_t,
		rho_t, v_x_t, v_y_t, v_z_t) = get_variables_at_snapshot(file_object)
		snapshot_array = create_snapshot_array(B_x_t, B_y_t, B_z_t, p_t,
												rho_t, v_x_t, v_y_t, v_z_t)
		X_t = split_snapshot_array(snapshot_array, grid_size)
		y_t = np.full((X_t.shape[0], ), t)

		x_observation_list.append(X_t)
		y_observation_list.append(y_t)

	X = np.concatenate(x_observation_list, axis = 0)
	y = np.concatenate(y_observation_list, axis = 0)
	return X, y

def variables_to_desgin_matrix(T, variables, grid_size = 5, periodic_bc = False):
	"""
	Create a design matrix ready for training machine learning models

	Parameters
	----------
	T : numpy.ndarray
		1D array containing all the time coordinates in the simulation
	variables : list of numpy.ndarray
		List of 4D numpy.ndarrays, each containing a variable from the simulati
	grid_size : numpy.ndarray
		The size of each grid to consider as an observation

	Returns
	-------
	X : numpy.ndarray
		5D array representing the design matrix
			The 0th axis specifies the observation
			The 1st axis represents the x-direction
			The 2nd axis represents the y-direction
			The 3rd axis represents the z-direction
			The 4th axis represents the 8 different predictor variables
	y : numpy.ndarray
		1D array representing the response vector
	"""
	x_observation_list= []
	y_observation_list = []
	for t in T:
		# Get the values of the variables at each point in time
		B_x_t = variables[0][:,:,:,t]
		B_y_t = variables[1][:,:,:,t]
		B_z_t = variables[2][:,:,:,t]
		p_t = variables[3][:,:,:,t]
		rho_t = variables[4][:,:,:,t]
		v_x_t = variables[5][:,:,:,t]
		v_y_t = variables[6][:,:,:,t]
		v_z_t = variables[7][:,:,:,t]
		snapshot_array = create_snapshot_array(B_x_t, B_y_t, B_z_t, p_t,
												rho_t, v_x_t, v_y_t, v_z_t)
		X_t = split_snapshot_array(snapshot_array, grid_size)
		y_t = np.full((X_t.shape[0], ), t)
		x_observation_list.append(X_t)
		y_observation_list.append(y_t)
	X = np.concatenate(x_observation_list, axis = 0)
	y = np.concatenate(y_observation_list, axis = 0)
	return X, y



