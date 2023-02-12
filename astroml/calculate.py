"""
Module for calculating quantities
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
		3D array containing the x coordinates of each point in simulation
	Y : numpy.ndarray
		3D array containing the y coordinates of each point in the simulation
	Z : numpy.ndarray
		3D array containing the z coordinates of each point in the simulation

	Returns
	-------
	cell_volume : float
		The volumme of each cell in the simulation
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
	B_x : scalar or numpy.ndarray
		The x component of the magnetic field
	B_y : scalar or numpy:ndarray
		The y component of the magnetic field
	B_z : scalar or numpy:ndarray
		The z component of the magnetic field

	Returns
	-------
	p_B : scalar or numpy.ndarray
		The magnetic energy density
	"""
	B_squared = B_x**2 + B_y**2 + B_z**2
	p_B = B_squared/(2*PERM_FREE_SPACE)
	return p_B

def magnetic_energy(cell_volume, p_B):
	"""
	Calculate the total magnetic energy

	Parameters
	----------
	cell_volume : float
		The volume of each cell in simulation
	p_B : numpy.ndarray
		4D array with the magnetic energy density

	Returns
	-------
	E_B : numpy.ndarray
		2D array with the magnetic energy at each moment in time in the
	simulation
	"""
	E_B = cell_volume*p_B.sum(axis = (0, 1, 2))
	return E_B

