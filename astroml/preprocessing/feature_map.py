"""
The feature_map module contains the function which returns a dictionary mapping
indices in the 5D numpy.array representing the matrix of features to the names
of the features
"""

def get_feature_map(features):
    """
    Return a dictionary mapping indicies to features

    Parameters
    ----------
    features : list of str
        The list of features to use

    Returns
    -------
    feature_map : dict of {int:str}
        Dictionary where the keys are the indicies of the 4th axis of the
        matrix of features and the values are the corresponding features
        at that index. The default is {0:"B_x",1:"B_y",2:"B_z"}
    """
    indices = range(len(features))
    feature_map = dict(zip(indices, features))
    return feature_map