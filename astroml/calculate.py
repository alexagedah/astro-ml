"""
The calculate module contains functions for calculating quantities for data
exploration.
"""
import scipy.constants as constants

# Constants
PERM_FREE_SPACE = constants.physical_constants["vacuum mag. permeability"][0]

def cell_volume(X, Y, Z):
	"""
	Return the volume of each cell in a simulation

	Parameters
	----------
	X : numpy.ndarray
		3D numyp.ndarray representing the x-coordinates of each point
	Y : numpy.ndarray
		3D numpy.ndarray representing the y-coordinates of each point
	Z : numpy.ndarray
		3D numpy.ndarray representing the z-coordinates of each point

	Returns
	-------
	cell_volume : float
		The volume of each cell in the simulation
	"""
	length_x = X[0,0,0] - X[0,1,0]
	length_y = Y[0,0,0] - Y[1,0,0]
	if Z.shape[2] == 1:
		length_z = 1
	else:
		length_z = Z[0,0,0] - Z[0,0,1]
	cell_volume = length_x*length_y*length_z
	return cell_volume

def magnetic_energy_density(B_x, B_y, B_z):
	"""
	Calculate the magnetic energy density

	Parameters
	----------
	B_x : numpy.ndarray
		4D numpy.ndarray representing the x-component of the magnetic field
	B_y : numpy.ndarray
		4D numpy.ndarray representing the y-component of the magnetic field
	B_z : numpy.ndarray
		4D numpy.ndarray representing the z-component of the magnetic field

	Returns
	-------
	p_B : numpy.ndarray
		4D numpy.ndarray representing the magnetic energy density
	"""
	B_squared = B_x**2 + B_y**2 + B_z**2
	p_B = B_squared/(2*PERM_FREE_SPACE)
	return p_B

def magnetic_energy(X, Y, Z, B_x, B_y, B_z):
	"""
	Calculate the total magnetic energy

	Parameters
	----------
	X : numpy.ndarray
		3D numyp.ndarray representing the x-coordinates of each point
	Y : numpy.ndarray
		3D numpy.ndarray representing the y-coordinates of each point
	Z : numpy.ndarray
		3D numpy.ndarray representing the z-coordinates of each point
	B_x : numpy.ndarray
		4D numpy.ndarray representing the x-component of the magnetic field
	B_y : numpy.ndarray
		4D numpy.ndarray representing the y-component of the magnetic field
	B_z : numpy.ndarray
		4D numpy.ndarray representing the z-component of the magnetic field

	Returns
	-------
	E_B : numpy.ndarray
		2D numpy.array representing the magnetic energy at each moment in time
	"""
	cell_volume = cell_volume(X, Y, Z)
	p_B = magnetic_energy_density(B_x, B_y, B_z)
	E_B = cell_volume*p_B.sum(axis = (0, 1, 2))
	return E_B

