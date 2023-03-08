"""
The simulation module contains the Simulation class.

There are two formats for storing data in numpy.ndarrays

Data Format One: Data Exploration
----------------------------------
Data is stored in multiple numpy.ndarrays array, and each array represents a
certain variable.

    * The 0th axis represents the x-axis
    * The 1st axis represents the y-axis
    * The 2nd axis represents the z-axis
    * The 4th axis represents the time axis

This format is ideal for data exploration and understanding how different
quantities for the disc (magnetic field, gas pressure, density, velocity etc...)
vary over space and time.

Format Two: Supervised Learning
-------------------------------
Data is stored in a 5D numpy.array representing a matrix of predictors and a
1D numpy.array represnting a vector of responses. For the matrix of predictors:

    * The 0th axis specifies the observation. Each observation is a small 
      snapshot of a disc at a certain moment in time. The size of the snapshot
      can be specified and will effect

        1. The total number of observations (larger snapshots means less
        observations)

        2. The length of the 1st, 2nd and 3rd axes (larger snapshots increase
        the size of these axes)

    * The 1st axis represents the x-direction
    * The 2nd axis represents the y-direction
    * The 3rd axis represents the z-direction
    * The 4th axis represents the different variables

        * Variable 0: x-component of the magnetic field
        * Variable 1: y-component of the magnetic field
        * Variable 2: z-component of the magnetic field
        * Variable 3: gas pressure
        * Variable 4: density
        * Variable 5: x-component of the fluid velocity
        * Variable 6: y-component of the fluid velocity
        * Variable 7: z-component of the fluid velocity


For the vector of responses, the 0th axis also specifies the observation.

This format is ideal for supervised learning.
"""
# 3rd Party
import numpy as np
# Local
from . import convert_format, read

class Simulation():
    """
    A simulation object represents a single accretion disc simulation.

    Parameters
    ----------
    data_relative_path : pathlib.Path
        A relative path to the folder containing the simulation data

    Attributes
    ----------
    x : numpy.ndaaray
        3D numpy.ndarray representing the x-coordiantes
    y : numpy.ndarray
        3D numpy.ndarray representing the y-coordiantes
    z : numpy.ndarray
        3D numpy.ndarray representing the z-coordiantes
    t : numpy.ndarray
        1D numpy.ndarray representing the time coordinates
    fluid_variables : dict of {str:numpy.ndarray}
        Dictionary containing all the fluid variables. The keys are strings which
        specify the fluid variable and the values are 4D numpy.ndarrays which
        represent the corresponding variable.

            * The 0th axis represents the x direction
            * The 1st axis represents the y direction
            * The 2nd axis represents the z direction
            * The 3rd axis represnets the time

    chi : float
        The magnetisation for the simulation

    Methods
    -------
    get_grid_observations(grid_size, response)
        Return a 5D numpy.ndarray representing a matrix of predictors and a 1D
        numpy.ndarray representing a vector of responses.
    plotting methods...

    calculate methods..
    """
    def __init__(self, data_relative_path):
        """
        TODO
        ----
        Constructor should take a sting, not a pathlib.Path object.
        """
        self.chi = float(data_relative_path.stem.replace("dot","."))
        self.x, self.y, self.z = read.get_cell_coordinates(data_relative_path)
        self.t = read.get_time_coordinates(data_relative_path)
        self.fluid_variables = read.get_fluid_variables(data_relative_path)

    def get_observations(self, observation_size, features, response):
        """
        Return a matrix of features and a vector of responses.

        Parameters
        ----------
        snapsht : int
            The length of each 2D grid or 3D cube
        features : list of str
            The list of features to use
        response : str
            The response variable

        Returns
        -------
        X : numpy.ndarray
            5D array representing the design matrix

                * The 0th axis specifies the observation
                * The 1st axis represents the x-direction
                * The 2nd axis represents the y-direction
                * The 3rd axis represents the z-direction
                * The 4th axis represents the 8 different predictors

        y : numpy.ndarray
            1D numpy.ndarray representing the response vector
        """
        X_t_list = []
        y_t_list = []
        for t in self.t:
            X_t = convert_format.disc_t_to_feature_matrix(self.fluid_variables,
                features,
                t,
                observation_size)
            X_t_list.append(X_t)
            if response == "time":
                y_t = np.full((X_t.shape[0], ), t)
            elif response == "chi":
                y_t = np.full((X_t.shape[0], ), self.chi)
            else:
                raise ValueError("Please choose time or chi as a response.")
            y_t_list.append(y_t)
        X = np.concatenate(X_t_list, axis = 0)
        y = np.concatenate(y_t_list, axis = 0)
        return X, y










