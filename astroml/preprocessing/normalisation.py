"""
The normaliastion module contains functions for normalising the data
"""
# 3rd Party
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

B_x_index = 0
B_y_index = 1
B_z_index = 2
v_x_index = 5
v_y_index = 6
v_z_index = 7

def component_normaliser(X, component_indices, magnitude_index):
    """
    Normalise 3 predictors, which are each component of a vector, by dividing
    the predictors by the magnitude of the vector

    Parameters
    ----------
    X : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors
    component_indices : list of int
        The locations along the 5th axis for the components of the vector
    magnitude_index : int
        The location along the 5th axis for the magnitude of the vector

    Returns
    -------
    numpy.ndarray
        5D numpy array representing the matrix of predictors where the 3 predictors
        specified by the 'component_indices' argument have been normalised
    """
    magnitude = X[:,:,:,:,magnitude_index]
    # Normalise the components of the velocity
    for i in component_indices:
        component = X[:,:,:,:,i]
        X[:,:,:,:,i] = np.divide(component,
            magnitude,
            out=np.zeros_like(component),
            where=magnitude!=0)
    return X

def full_standard_scaler(X_train, X_valid, X_test):
    """
    Standardise all the predictors

    Parameters
    ---------
    X_train : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the
        training data set
    X_valid : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the
        validation data set
    X_test : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the
        test data set
    Returns
    -------
    X_train_scaled : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the training
        data set where all the predictors have been standardised
    X_valid_scaled : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the validation
        data set where all the predictors have been standardised
    X_test_scaled : numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the test
        data set where all the predictors have been standardised
    """
    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(X_train)
    X_train_scaled = norm_layer(X_train)
    X_valid_scaled = norm_layer(X_valid)
    X_test_scaled = norm_layer(X_test)
    return X_train_scaled, X_valid_scaled, X_test_scaled

def physical_standard_scaler(X_train, X_valid, X_test):
    """
    Standardise all the predictors that aren't components of a vector

    This function standardises all the predictors except the components of the
    magnetic field and the components of the velocity

    Parameters
    ---------
    X_train : numpy.ndarray
        4D o5D numpy.ndarray representing the matrix of predictors for the
        training data set
    X_valid : numpy.ndarray
        4D or 5D numpy.ndarray representing the matrix of predictors for the
        validation data set
    X_test : numpy.ndarray
        4D or 5D numpy.ndarray representing the matrix of predictors for the
        test data set
    Returns
    -------
    numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the training
        data set where all the predictors have been standardised
    numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the validation
        data set where all the predictors have been standardised
    numpy.ndarray
        5D numpy.ndarray representing the matrix of predictors for the test
        data set where all the predictors have been standardised
    """
    # For each variable
    for variable in range(X_train.shape[4]):
        # If the variable is not a component of a vector
        if variable not in [B_x_index, B_y_index, B_z_index, v_x_index, v_y_index, v_z_index]:
            # Scale the variable
            training_mean = X_train[:,:,:,:,variable].mean()
            training_std = X_train[:,:,:,:,variable].std(ddof=1)
            X_train[:,:,:,:,variable] = standard_scaler(X_train[:,:,:,:,variable],
                training_mean, training_std)
            X_valid[:,:,:,:,variable] = standard_scaler(X_valid[:,:,:,:,variable],
                training_mean, training_std)
            X_test[:,:,:,:,variable] = standard_scaler(X_test[:,:,:,:,variable],
                training_mean, training_std)
    return X_train, X_valid, X_test

def min_max_scaler(y_train, y_valid, y_test):
    """
    Apply min-max scaling to the vector of response

    Parameters
    ---------
    y_train : numpy.ndarray
        1D numpy.ndarray representing the vector of response for the training
        data set
    y_valid : numpy.ndarray
        1D numpy.ndarray representing the vector of response for the validation
        data set
    y_test : numpy.ndarray
        1D numpy.ndarray representing the vector of response for the test
        data set

    Returns
    -------
    y_train_scaled : numpy.ndarray
    y_valid_scaled : numpy.ndarray
    y_test_scaled : numpy.ndarray
    min_max_transformer : sklearn.preprocessing.MinMaxScaler
        The transformer for min-max scaling
    """
    min_max_transformer = MinMaxScaler(feature_range=(-1,1))
    y_train_scaled = min_max_transformer.fit_transform(y_train)
    y_valid_scaled = min_max_transformer.transform(y_valid)
    y_test_scaled = min_max_transformer.transform(y_test)
    return y_train_scaled, y_valid_scaled, y_test_scaled, min_max_transformer

def standard_scaler(x, mean, std):
    return (x - mean)/std



