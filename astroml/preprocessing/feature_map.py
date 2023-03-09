"""
The feature_map module contains the function which returns a dictionary mapping
indices in the 5D numpy.array representing the matrix of features to the names
of the features
"""
def verbose_features(features):
    """
    Extend the list of features to the verbose version.

    This function extends the list of features the verbose version.
    If "default" is included, then this is replaced with all the default fluid,
    variables: "B_x","B_y","B_z","p","rho","u_x","u_y","u_z"

    Parameters
    ----------
    features : list of str
        The list of features to use

    Returns
    -------
    verbose_features : list of str
        The verbose list of features

    """
    verbose_features = []
    for feature in features:
        if feature == "default":
            verbose_features += ["B_x","B_y","B_z","p","rho","u_x","u_y","u_z"]
        elif feature == "2d":
            verbose_features += ["B_x","B_y","p","rho","u_x","u_y"]
        else:
            verbose_features.append(feature)
    return verbose_features

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
    verbose_features_list = verbose_features(features)
    indices = range(len(verbose_features_list))
    feature_map = dict(zip(indices, verbose_features_list))
    return feature_map

