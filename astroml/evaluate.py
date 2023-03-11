"""
The evaluate module contains functions for evaluating artificial neural networks.
"""
# Standard Libary
import pathlib
import os
# 3rd Party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
# Local
from . import plot

mpl.rcParams["font.family"] = "Times New Roman"
plt.style.use("default")

def time_predictions(y, y_pred):
    """
    Return the predicted times by the model

    Parameters
    ----------
    y : numpy.ndarray
        numpy.ndarray representing the observed vector of responses
    y_pred : numpy.ndarray
        numpy.ndarray representing the predicted vector of responses by a fitted
        model

    Returns
    -------
    numpy.ndarray
        numpy.ndarray representing the fitted model's predictions, after rounding
        the predictions to the nearest time
    """
    max_time = y.max()
    min_time = y.min()
    y_pred[y_pred>max_time] = max_time
    y_pred[y_pred<min_time] = min_time
    return y_pred.round().astype(np.int64)

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
    y_pred_adjusted = time_predictions(y, y_pred)
    accuracy = accuracy_score(y.round(), y_pred_adjusted)
    return accuracy

def chi_responses(y):
    """
    Return the transformed values of the response so that they are all integers

    Parameters
    ---------
    y : numpy.ndarray
        numpy.ndarray representing the vector of responses

    Returns
    -------
    numpy.ndarray
        numpy.ndrray represening the integer transformed vector of responses
    """
    y = np.where(y > 0.125, 2, y)
    y = np.where(y < 0.075, 0, y)
    y = np.where((y < 2) & (y > 0), 1, y)
    return y

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
    y = chi_responses(y)
    y_pred = chi_predictions(y_pred)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def get_confusion_matrix(y, y_pred, response):
    """
    Plot a confusion matrix

    Parameters
    ----------
    y : numpy.ndarray
    y_pred : numpy.ndarray
    response : str

    Returns
    -------
    arr : numpy.ndarray
        numpy.ndarray representing the confusion matrix
    """
    if response == "time":
        y = y.round()
        y_pred = time_predictions(y, y_pred)
    elif response == "chi":
        y = chi_responses(y)
        y_pred = chi_responses(y_pred)
    classes = sorted(np.unique(y))
    arr = confusion_matrix(y, y_pred, labels = classes)
    return arr

def plot_confusion_matrix(y, y_pred, response, model_name):
    """
    Plot a confusion matrix

    Parameters
    ----------
    y : numpy.ndarray
    y_pred : numpy.ndarray
    response : str
    """
    if response == "time":
        og_classes = sorted(np.unique(y.round()))
    elif response == "chi":
        og_classes = sorted(np.unique(y))
    arr = get_confusion_matrix(y, y_pred, response)
    disp = ConfusionMatrixDisplay(confusion_matrix=arr,
        display_labels=og_classes)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    disp.plot(ax=ax1, 
        include_values=False,
        xticks_rotation="vertical")
    fig.savefig(f"results/{model_name}_confusion")

def evaluate(model, X_train, X_valid, y_train_scaled, y_valid_scaled):
    """
    Return the training and validation loss of a model

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

    Returns
    -------
    training_loss : float
        The loss calculated on the training data set
    validation_loss : float
        The loss calculated on the validation data set
    """
    training_loss = model.evaluate(X_train, y_train_scaled)
    validation_loss = model.evaluate(X_valid, y_valid_scaled)
    return training_loss, validation_loss

def accuracy(y, y_pred, response):
    """
    Parameters
    ----------
    y : numpy.ndarray
        numpy.ndarray representing the observed vector of responses
    y_pred : numpy.ndarray
        numpy.ndarray representing the predicted vector of responses by a fitted
        model
    response : str
        The name of the response variable

    Returns
    -------
    accuracy : float
    """
    if response == "time":
        accuracy = time_accuracy(y, y_pred)
    elif response == "chi":
        accuracy = chi_accuracy(y, y_pred)
    return accuracy

def evaluate_model(model, model_name, X_train, X_valid, y_train_scaled, y_valid_scaled, response, min_max_scaler):
    """
    Evaluate the accuracy of the model on the training data and validation data

    Parameters
    ----------
    model : tf.keras.Models
        The model to evaluate
    model_name : str
        The name of the model to evaluate
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
        The name of the response variable. options are “time” and “chi”
    min_max_scaler : sklearn.preprocessing.MinMaxScaler
        A fitted transformer for transforming the response

    Returns
    -------
    results : pandas.Series
        pandas.Series with the training and validation mean squared error and
        accuracy of the model
    """
    training_mse, validation_mse = evaluate(model, X_train, X_valid, y_train_scaled, y_valid_scaled)
    y_train = min_max_scaler.inverse_transform(y_train_scaled)
    y_valid = min_max_scaler.inverse_transform(y_valid_scaled)
    y_train_pred = min_max_scaler.inverse_transform(model.predict(X_train))
    y_valid_pred = min_max_scaler.inverse_transform(model.predict(X_valid))
    training_accuracy = accuracy(y_train, y_train_pred, response)
    validation_accuracy = accuracy(y_valid, y_valid_pred, response)
    results = pd.Series({
    "Training MSE":training_mse,
    "Validation MSE":validation_mse,
    "Training Accuracy":training_accuracy,
    "Validation Accuracy":validation_accuracy},
    name="Results")
    plot_confusion_matrix(y_valid, y_valid_pred, response, model_name)
    results.to_csv(f"results/{model_name}")
    return None





