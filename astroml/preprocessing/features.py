"""
The features module contains functions for creating and adding features
"""
# 3rd Party
import numpy as np
# Local
from . import normalisation

B_x_INDEX = 0
B_y_INDEX = 1
B_z_INDEX = 2
p_INDEX = 3
rho_INDEX = 4
v_x_INDEX = 5
v_y_INDEX = 6
v_z_INDEX = 7
B_INDEX = 8
v_INDEX = 9

def add_feature(X, new_feature):
    """
    Add a feature to the matrix of predictors

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

            * The 0th axis specifies the observation
            * The 1st axis represents the x-direction
            * The 2nd axis represents the y-direction
            * The 3rd axis represents the z-direction
            * The 4th axis represents the different features/predictor variables

    new_feature numpy.ndarray
        4D numpy.ndarray representing the new feature

            * The 0th axis specifies the observation
            * The 1st axis represents the x-direction
            * The 2nd axis represents the y-direction
            * The 3rd axis represents the z-direction
            * The 4th axis represents the different predictor variables
    Returns
    -------
    X_with_new_feature : numpy.ndarray
        5D numpy array representing the matrix of predictors with the new feature
        added
    """
    X_with_new_feature = np.concatenate([X, new_feature[:,:,:,:,np.newaxis]], 4)
    return X_with_new_feature

def add_magnetic_field_magnitude(X):
    """
    Add the magnitude of the magnetic field to the matrix of predictors

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_with_B : numpy.ndarray
        5D numpy array representing the matrix of predictors with the magnitude
        of the magnetic field included as a predictor.
    """
    B = np.sqrt(X[:,:,:,:,B_x_INDEX]**2 + X[:,:,:,:,B_y_INDEX]**2 + X[:,:,:,:,B_z_INDEX]**2)
    X_with_B = add_feature(X, B)
    return X_with_B

def add_speed(X):
    """
    Add the speed of the fluid to the matrix of predictors

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_with_speed : numpy.ndarray
        5D numpy array representing the matrix of predictors with the speed
        of the fluid included
    """
    speed = np.sqrt(X[:,:,:,:,v_x_INDEX]**2 + X[:,:,:,:,v_y_INDEX]**2 + X[:,:,:,:,v_z_INDEX]**2)
    X_with_speed = add_feature(X, speed)
    return X_with_speed

def add_plasma_beta(X):
    """
    Add plasma beta as a feature

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_transformed : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors with the plasma
    beta added as a predictor.
    """
    plasma_beta = X[:,:,:,:,p_INDEX]/(X[:,:,:,:,B_INDEX]**2)
    X_transformed = add_feature(X, plasma_beta)
    return X_transformed

EXTRA_FEATURES = {
"B":add_magnetic_field_magnitude,
"v":add_speed,
"beta":add_plasma_beta
}

def add_features(X, features):
    """
    Add features to the matrix of predictors

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors
    features : list of str
        A list of the features to add

    Returns
    -------
    numpy.ndarray
        numpy.ndarray representing the matrix of predictors with all the new
        predictors added
    """
    for feature in features:
        X = EXTRA_FEATURES[feature](X)
    return X
