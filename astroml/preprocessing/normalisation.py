"""
The normaliastion module contains functions for normalising the data
"""
# 3rd Party
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class NoneTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X

def standard_scaler(X_train, X_valid, X_test):
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
    training_means = X_train.mean(axis=0)
    training_stds = X_train.std(axis=0, ddof=1)
    X_train_scaled = standardise(X_train, training_means, training_stds)
    X_valid_scaled = standardise(X_valid, training_means, training_stds)
    X_test_scaled = standardise(X_test, training_means, training_stds)
    return X_train_scaled, X_valid_scaled, X_test_scaled

def none_transformer(y_train, y_valid, y_test):
    """
    Apply no scaling to the vector of responses

    Returns
    -------
    y_train_scaled : numpy.ndarray
    y_valid_scaled : numpy.ndarray
    y_test_scaled : numpy.ndarray
    none_transformer : sklearn.preprocessing.MinMaxScaler
    """
    none_transformer = NoneTransformer()
    y_train_scaled = none_transformer.fit_transform(y_train)
    y_valid_scaled = none_transformer.transform(y_valid)
    y_test_scaled = none_transformer.transform(y_test)
    return y_train, y_valid, y_test, none_transformer

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

def standardise(x, mean, std):
    """
    Standardise a feature

    Parameters
    ----------
    x : numpy.ndarray
    mean : float
    std : float
    Returns
    -------
    x_standarised : numpy.ndarray
    """
    x_minus_mean = x - mean
    x_standarised = np.divide(x_minus_mean, std, out=np.zeros_like(x_minus_mean), where=std!=0)
    return x_standarised





