"""
The dimension reduction module contains the function for removing the z-axis
(3rd axis) from each observation if the data is 2D, and the predictors B_z and
v_z
"""
import numpy as np

def remove_extra_dimesions(X):
    """
    If the observations are 2D, removes the z axis from each observation and
    removes the B_z and v_z predictor variables

    Parameters
    ----------
    X : numpy.ndarray
        5D array representing the matrix of predictors

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

    """
    if X.shape[3] == 1:
        # Remove the z axis, b_z
        X = np.squeeze(X, axis = 3)
        # Remove B_z and v_z
        return np.delete(X, [2, 7], axis = 3)
    else:
        return X

def remove_extra_dimensions_all(X_train, X_valid, X_test):
    """
    """
    X_train_transformed = remove_extra_dimesions(X_train)
    X_valid_transformed = remove_extra_dimesions(X_valid)
    X_test_transformed = remove_extra_dimesions(X_test)
    return X_train_transformed, X_valid_transformed, X_test_transformed
