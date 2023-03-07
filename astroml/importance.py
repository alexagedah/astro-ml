"""
The importance module contains functions for understanding the importance of
each feature
"""
# Standard Library
import copy
# 3rd Party
import pandas as pd
import numpy as np

def shuffle_feature(X, feature_index):
    """
    Return a shuffled matrix of predictors

    Parameters
    ----------
    X : numpy.ndarray
        4D or 5D numpy.ndarray representing the matrix of features
    feature_index : numpy.ndarray
        The index for the feature to shuffle
    Returns
    -------
    X_shuffled : numpy.ndarray
        4D or 5D numpy.ndarray representing the matrix of features where one
        of the features have been shuffled
    """
    X_shuffled = X.copy()
    if X.ndim == 4:
        np.random.shuffle(X_shuffled[:,:,:,feature_index])
    elif X.ndim == 5:
        np.random.shuffle(X_shuffled[:,:,:,:,feature_index])
    return X_shuffled

def feature_importance(model, model_name, X_valid, y_valid):
    """
    Return a DataFrame with importance of each feature

    Parameters
    ----------
    model : tensorflow.keras.Models
        The model to do feature importance for
    model_name : str
        The name of the model
    X_valid : numpy.ndarray
        4D or 5D numpy.ndarray representing the matrix of features for the
        validation data set
    y_valid :
        1D numpy.ndarray representing the vector of responses for the validation
        data set

    Returns
    -------
    importance_series : pandas.Series
        pandas.Series with the validation mean squared error of the model when
        each feature is permuted. The most important features has the % increase
        in the mean squared error.
    """
    data = []
    for feature in range(X_valid.shape[-1]):
        X_valid_shuffled = shuffle_feature(X_valid, feature)
        shuffled_mse = model.evaluate(X_valid_shuffled, y_valid)
        feature_df = pd.Series(shuffled_mse,
            index = [feature],
            name = "Importance")
        data.append(feature_df)
    validation_mse = model.evaluate(X_valid, y_valid)
    importance_series = (pd.concat(data, axis = 0) - validation_mse)/validation_mse
    importance_series.sort_values(ascending=False, inplace = True)
    importance_series.to_csv(f"importance/{model_name}")
    return importance_series



