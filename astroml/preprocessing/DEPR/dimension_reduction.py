"""
DEPRECIATED
The dimension reduction module contains the function for removing the z-axis
(3rd axis) from each observation if the data is 2D, and the predictors B_z and
v_z
"""
import numpy as np

def remove_feature_map_dimensions(feature_map):
    """
    Remove B_z and v_z for the feature map and correct the indices

    Parameters
    ----------
    feature_map : dictionary of {int:str}
        Dictionary indicating which features are at which index in the matrix of
        predictors

    Returns
    -------
    adjusted_feature_map : dictionary of {int:str}
        Dictionary indicating which features are at which index in the matrix of
        predictors, adjusted for the dimension reduction
    """
    adjusted_feature_map = {}
    for index, feature in feature_map.items():
        # Make no adjustment to B_x and B_y
        if index < 2:
            adjusted_feature_map[index] = feature
        # Reduce index of p, rho, v_x and v_y by 1
        elif index >=3 and index < 7:
            adjusted_feature_map[index-1] = feature
        # Reduce index of everyting after v_z by 2
        elif index >= 8:
            adjusted_feature_map[index-2] = feature
    return adjusted_feature_map

def remove_extra_dimesions(X, feature_map):
    """
    If the observations are 2D, removes the z axis from each observation and
    removes the B_z and v_z predictor variables

    Parameters
    ----------
    X : numpy.ndarray
        5D array representing the matrix of predictors
    feature_map : dictionary of {int:str}
        Dictionary indicating which features are at which index in the matrix of
        predictors

    Returns
    -------
    numpy.ndarray
        4D or 5D numpy.narray. If the simulation data is 2D, the numpy.ndarray
        is as follows:

            * The 0th axis specifies the observation
            * The 1st axis represents the x-direction
            * The 2nd axis represents the y-direction
            * The 3rd axis represents the z-direction
            * The 4th axis represents the different predictor variables

        If the simulation data is 3D, the 3rd and 4th axis are slightly different.
        numpy.ndarray

            * The 3rd axis represents the different predictor variables, but the
            z-component of the magnetic field and the z-component of the velocity
            are dropped as predictors
            * There is no 4th axis. The matrix of predictors has 4 dimenions.

    adjusted_feature_map : dictionary of {int:str}
        Dictionary indicating which features are at which index in the matrix of
        predictors, adjusted for the dimension reduction
    """
    if X.shape[3] == 1:
        adjusted_feature_map = remove_feature_map_dimensions(feature_map)
        # Remove the z axis, b_z
        X = np.squeeze(X, axis = 3)
        # Remove B_z and v_z
        return np.delete(X, [2, 7], axis = 3), adjusted_feature_map
    else:
        return X, feature_map

