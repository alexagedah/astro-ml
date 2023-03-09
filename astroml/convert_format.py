"""
The convert_format module contains functions for converting data from the disc
format to the machine learning format.

Format One : Disc Format
------------------------

In the disc format, fluid variables are stored in numpy.ndarrays array, and
numpy.ndarray.

    * The 0th axis represents the x-axis
    * The 1st axis represents the y-axis
    * The 2nd axis represents the z-axis
    * The 4th axis represents the time axis

This format is ideal for data exploration and understanding how different
quantities for the disc (magnetic field, gas pressure, density, velocity etc...)
vary over space and time.

Format Two: Machine Learning
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

This format is ideal for machine learning.
"""
import numpy as np

def get_features_at_time(all_fluid_variables, features_to_include, time):
    """
    Return a numpy.ndarray representing features at given time

    This function returns a 4D numpy.ndarray which represents the features
    at a moment in time. Note that the format the the 4D numpy.ndarray is different
    to the format disc format.

        * The 0th axis represents the x-direction
        * The 1st axis represents the y-direction
        * The 2nd axis represents the z-direction
        * The 3rd axis represents the features

    This format is also not quite yet in the machine learning format.

    Parameters
    ----------
    all_fluid_variables : dict of {str:numpy.ndarray}
        Dictionary containing all the fluid variables. The keys are strings which
        specify the fluid variable and the values are 4D numpy.ndarrays which
        represent the corresponding variable.
    features_to_include : list of str
        A list of the fluid variables to return
    time : The time to return the fluid variables for

    Returns
    -------
    features_t : numpy.ndarray
        4D numpy.ndarray representing the features at a certain moment in time
    """
    features_list_t = []
    for feature_name in features_to_include:
        feature_t = all_fluid_variables[feature_name][:,:,:,time]
        features_list_t.append(feature_t[:,:,:,np.newaxis])
    features_t = np.concatenate(features_list_t, axis = 3)
    return features_t

def divide_disc(features_t, observation_size):
    """
    Divide the disc into observations to create a matrix of features

    Parameters
    ----------
    features_t : numpy.ndarray
        4D numpy.ndarray representing the features at a certain moment in time

            * The 0th axis represents the x-direction
            * The 1st axis represents the y-direction
            * The 2nd axis represents the z-direction
            * The 3rd axis represents the features

    observation_size : int
        The size of each grid/cube to consider as an observation

    Returns
    -------
    X_t : numpy.ndarray
        5D numpy.ndarray representing a matrix of features

    TODO
    ----
    UNIT TESTING
        There should be certain relationships between different observations!
        Have tests to make sure the array has been formed correctly
    BOUNDARY CONDITIONS
        Be able to specifiy if you want to use periodic boundary conditions
        This means we can get a few extra observations
    """
    X_t_list = []
    x_cells = features_t.shape[0]
    y_cells = features_t.shape[1]
    z_cells = features_t.shape[2]
    for x_start in range(0, x_cells):
        x_end = x_start + observation_size
        if x_end == x_cells:
            continue
        elif x_end > x_cells:
            continue

        for y_start in range(0, y_cells):
            y_end = y_start + observation_size
            if y_end == y_cells:
                continue
            elif y_end > y_cells:
                continue

            for z_start in range(0, z_cells):
                if z_cells == 1:
                    observation = np.expand_dims(
                        features_t[x_start:x_end,y_start:y_end,:,:],
                    0
                    )
                    X_t_list.append(observation)
    X_t = np.concatenate(X_t_list, axis = 0)
    return X_t

def disc_t_to_feature_matrix(all_fluid_variables, features_to_include, time, observation_size):
    """
    Transform fluid variables at a time to a matrix of features for that time

    Parameters
    ----------
    all_fluid_variables : dict of {str:numpy.ndarray}
        Dictionary containing all the fluid variables. The keys are strings which
        specify the fluid variable and the values are 4D numpy.ndarrays which
        represent the corresponding variable.
    features_to_include : list of str
        A list of the fluid variables to return
    time : The time to return the fluid variables for

    Returns
    -------
    X_t : numpy.ndarray
        5D numpy.ndarray representing a matrix of features

    """
    features_t = get_features_at_time(all_fluid_variables, features_to_include, time)
    X_t = divide_disc(features_t, observation_size)
    return X_t
