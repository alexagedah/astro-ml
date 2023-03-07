"""
The test module contains functions for testing artificial neural networks.
"""
# Standard Libary
import pathlib
import os
# 3rd Party
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
# Libary Specific
from . import plot

def time_accuracy(y, y_pred):
    """
    Evaluate the accuracy of a model for predicting the time of an observation
    on the training data and validation data

    Parameters
    ----------
    y : numpy.ndarray
        numpy.ndarray representing the observed vector of responses
    y_pred : numpy.ndarray
        numpy.ndarray representing the predicted vector of responses by a fitted
        model

    Returns
    -------
    accuracy : float
    """
    max_time = y.max()
    min_time = y.min()
    y_pred[y_pred>max_time] = max_time
    y_pred[y_pred<min_time] = min_time
    accuracy = accuracy_score(y.round(), y_pred.round())
    return accuracy

def chi_accuracy(y, y_pred):
    """
    Evaluate the accuracy of a model for predicting the magnetisation of an
    observation on the training data and validation data

    Parameters
    ----------
    y : numpy.ndarray
        numpy.ndarray representing the observed vector of responses
    y_pred : numpy.ndarray
        numpy.ndarray representing the predicted vector of responses by a fitted
        model

    Returns
    -------
    accuracy : float
    """
    y = np.where(y > 0.125, 2, y)
    y = np.where(y < 0.075, 0, y)
    y = np.where((y < 2) & (y > 0), 1, y)
    y_pred = np.where(y_pred > 0.125, 2, y_pred)
    y_pred = np.where(y_pred < 0.075, 0, y_pred)
    y_pred = np.where((y_pred < 2) & (y_pred > 0), 1, y_pred)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def evaluate_model(model, X_train, X_valid, y_train_scaled, y_valid_scaled, response, min_max_scaler):
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
    y_train_scaled : numpy.ndarray
        numpy.ndarray representing the vector of responses for the training
        data set
    y_valid_scaled : numpy.ndarray
        numpy.ndarray representing the vector of responses for the validation
        data set
    response : str
        The name of the response variable. Options are “time” and “chi”
    min_max_scaler : sklearn.preprocessing.MinMaxScaler
        A fitted transformer for transforming the response
    """
    print("Mean Squared Error: ")
    print("Training: ")
    print(model.evaluate(X_train, y_train_scaled))
    print("Validation: ")
    print(model.evaluate(X_valid, y_valid_scaled))
    y_train = min_max_scaler.inverse_transform(y_train_scaled)
    y_valid = min_max_scaler.inverse_transform(y_valid_scaled)
    y_train_pred = min_max_scaler.inverse_transform(model.predict(X_train))
    y_valid_pred = min_max_scaler.inverse_transform(model.predict(X_valid))
    if response == "time":
        training_accuracy = time_accuracy(y_train, y_train_pred)
        validation_accuracy = time_accuracy(y_valid, y_valid_pred)
    elif response == "chi":
        training_accuracy = chi_accuracy(y_train, y_train_pred)
        validation_accuracy = chi_accuracy(y_valid, y_valid_pred)
    print("\nAccuracy: ")
    print(f"Training accuracy: {training_accuracy}")
    print(f"Validation accuracy: {validation_accuracy}")
