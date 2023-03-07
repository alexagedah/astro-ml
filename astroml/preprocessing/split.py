"""
The split module contains functions for splitting the data set into a training data
set, a validation data set and a test data set
"""
from sklearn.model_selection import train_test_split

def train_valid_test_split(X, y, train_size, valid_size):
    """
    Split a data set into a training set, a validation set and a test set

    Parameters
    ----------
    X : numpy.ndarray
        The matrix of predictors
    y : numpy.ndarray
        The vector of responses
    train_size : float
        The proportion of the data set to put into the training set
    valid_size : float
        The proportion of the data set to put into the validation set

    Returns
    -------
    X_train : numpy.ndarray
        The matrix of predictors for the training data set
    X_valid : numpy.ndarray
        The matrix of predictors for the validation data set
    X_test : numpy.ndarray
        The matrix of predictors for the test data set
    y_train : numpy.ndarray
        The vector of responses for the training data set
    y_valid : numpy.ndarray
        The vector of responses for the validation data set
    y_test : numpy.ndarray
        The vector of responses for the test data set
    """
    test_size = 1 - train_size - valid_size
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X,
        y,
        test_size=valid_size+test_size,
        stratify=y,
        random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test,
                    y_valid_test,
                    test_size = (test_size/(test_size+valid_size)),
                    stratify = y_valid_test,
                    random_state = 42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test