"""
The test module contains functions for testing artificial neural networks.
"""
# Standard Libary
import pathlib
import os
# 3rd Party
from sklearn.metrics import accuracy_score
import tensorflow as tf
# Libary Specific
from . import plot

def evaluate_time_model(model, X_train, X_valid, y_train, y_valid, min_max_scaler):
    """
    Evaluate the accuracy of a model for predicting the time of an observation
    on the training data and validation data

    Parameters
    ----------
    model : tf.keras.Models
        The model to evaluate
    X_train : numpy.ndarray
        numpy.ndarray representing the matrix of predictors for the training
        data set
    X_valid : numpy.ndarray
        numpy.ndarray representing the matrix of predictors for the validation
        data set
    y_train : numpy.ndarray
        numpy.ndarray representing the vector of responses for the training
        data set
    y_valid : numpy.ndarray
        numpy.ndarray representing the vector of responses for the validation
        data set
    min_max_scaler : sklearn.preprocessing.MinMaxScaler
        A fitted transformer for transforming the response

    Returns
    -------
    training_accuracy : float
    validation_accuracy : float
    """
    y_train = min_max_scaler.inverse_transform(y_train)
    y_valid = min_max_scaler.inverse_transform(y_valid)
    y_train_pred = min_max_scaler.inverse_transform(model.predict(X_train))
    y_valid_pred = min_max_scaler.inverse_transform(model.predict(X_valid))
    print(y_train)
    print(y_train.shape)
    print(y_train_pred.shape)
    training_accuracy = accuracy_score(y_train.round(), y_train_pred.round())
    validation_accuracy = accuracy_score(y_valid.round(), y_valid_pred.round())
    return training_accuracy, validation_accuracy

def evaluate_chi_model(model, X_train, X_valid, y_train, y_valid, min_max_scaler):
    """
    Evaluate the accuracy of a model for predicting the time of an observation
    on the training data and validation data

    Parameters
    ----------
    model : tf.keras.Models
        The model to evaluate
    X_train : numpy.ndarray
        numpy.ndarray representing the matrix of predictors for the training
        data set
    X_valid : numpy.ndarray
        numpy.ndarray representing the matrix of predictors for the validation
        data set
    y_train : numpy.ndarray
        numpy.ndarray representing the vector of responses for the training
        data set
    y_valid : numpy.ndarray
        numpy.ndarray representing the vector of responses for the validation
        data set
    min_max_scaler : sklearn.preprocessing.MinMaxScaler
        A fitted transformer for transforming the response

    Returns
    -------
    training_accuracy : float
    validation_accuracy : float
    """
    pass
    return training_accuracy, validation_accuracy

def evaluate_model(model, X_train, X_valid, y_train, y_valid, response, min_max_scaler):
    """
    Evaluate the accuracy of the model on the training data and validation data

    Parameters
    ----------
    model : tf.keras.Models
        The model to evaluate
    X_train : numpy.ndarray
        numpy.ndarray representing the matrix of predictors for the training
        data set
    X_valid : numpy.ndarray
        numpy.ndarray representing the matrix of predictors for the validation
        data set
    y_train : numpy.ndarray
        numpy.ndarray representing the vector of responses for the training
        data set
    y_valid : numpy.ndarray
        numpy.ndarray representing the vector of responses for the validation
        data set
    response : str
        The name of the response variable. Options are “time” and “chi”
    min_max_scaler : sklearn.preprocessing.MinMaxScaler
        A fitted transformer for transforming the response
    """
    if response == "time":
        training_accuracy, validation_accuracy = evaluate_time_model(model, X_train, X_valid, y_train, y_valid, min_max_scaler)
    elif response == "chi":
        training_accuracy, validation_accuracy = evaluate_chi_model(model, X_train, X_valid, y_train, y_valid, min_max_scaler)
    print(f"Training accuracy: {training_accuracy}")
    print(f"Validation accuracy: {validation_accuracy}")
