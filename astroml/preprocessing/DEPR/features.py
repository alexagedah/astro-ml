"""
DEPRECIATED
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

DEFAULT_FEATURE_MAP = {
B_x_INDEX:"B_x",
B_y_INDEX:"B_y",
B_z_INDEX:"B_z",
p_INDEX:"p",
rho_INDEX:"rho",
v_x_INDEX:"v_x",
v_y_INDEX:"v_y",
v_z_INDEX:"v_z"
}

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

def add_magnetic_energy_denstiy(X):
    """
    Add the magnetic energy denstiy to the matrix of predictors

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_with_E_m : numpy.ndarray
        5D numpy array representing the matrix of predictors with the magnetic
        energy density included as a predictor.
    """
    E_m = X[:,:,:,:,B_x_INDEX]**2 + X[:,:,:,:,B_y_INDEX]**2 + X[:,:,:,:,B_z_INDEX]
    X_with_E_m = add_feature(X, E_m)
    return X_with_E_m

def add_speed_squared(X):
    """
    Add the speed of the fluid to the matrix of predictors

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_with_speed_squared : numpy.ndarray
        5D numpy array representing the matrix of predictors with the squared 
        speed of the fluid included
    """
    speed_squared = X[:,:,:,:,v_x_INDEX]**2 + X[:,:,:,:,v_y_INDEX]**2 + X[:,:,:,:,v_z_INDEX]**2
    X_with_speed = add_feature(X, speed_squared)
    return X_with_speed_squared

def add_plasma_beta(X):
    """
    Add plasma beta as a feature

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_with_plasma_beta : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors with the plasma
    beta added as a predictor.
    """
    E_m = X[:,:,:,:,B_x_INDEX]**2 + X[:,:,:,:,B_y_INDEX]**2 + X[:,:,:,:,B_z_INDEX]
    plasma_beta = X[:,:,:,:,p_INDEX]/E_m
    X_with_plasma_beta = add_feature(X, plasma_beta)
    return X_with_plasma_beta

def add_speed_of_sound(X):
    """
    Add speed of sound

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_with_c_s : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors with the speed
    of sound added as a predictor
    """
    c_s = X[:,:,:,:,p_INDEX]/X[:,:,:,:,rho_INDEX]
    X_with_c_s = add_feature(X, c_s)
    return X_with_c_s

def add_alfven_wave_speed(X):
    """
    Add alfven wave speed as 

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_with_v_A : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors with the Alfven
    wave speed added as a predictor
    """
    B = np.sqrt(X[:,:,:,:,B_x_INDEX]**2 + X[:,:,:,:,B_y_INDEX]**2 + X[:,:,:,:,B_z_INDEX]**2)
    v_A = B/np.sqrt(X[:,:,:,:,rho_INDEX])
    X_with_v_A = add_feature(X, v_A)
    return X_with_v_A

def add_cross_helicity(X):
    """
    Add cross-helicty density as a predictor

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors

    Returns
    -------
    X_with_v_dot_B : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors with the cross-helicty
    wave speed added as a predictor
    """
    v_dot_B = (X[:,:,:,:,B_x_INDEX]*X[:,:,:,:,v_x_INDEX] + 
        X[:,:,:,:,B_y_INDEX]*X[:,:,:,:,v_y_INDEX] + 
        X[:,:,:,:,B_z_INDEX]*X[:,:,:,:,v_z_INDEX])
    X_with_v_dot_B = add_feature(X, v_dot_B)
    return X_with_v_dot_B

EXTRA_FEATURE_FUNCTIONS = {
"B":add_magnetic_field_magnitude,
"v":add_speed,
"E_m":add_magnetic_energy_denstiy,
"v_squared":add_speed_squared,
"beta":add_plasma_beta,
"c_s":add_speed_of_sound,
"v_A":add_alfven_wave_speed,
"v_dot_B":add_cross_helicity
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
    feature_map : dictionary of {int:str}
        Dictionary indicating which features are at which index in the matrix of
        predictors
    """
    extra_feature_map = {}
    for i, feature in enumerate(features):
        extra_feature_map[8+i] = feature
        X = EXTRA_FEATURE_FUNCTIONS[feature](X)
    feature_map = DEFAULT_FEATURE_MAP.copy()
    feature_map.update(extra_feature_map)
    return X, feature_map
