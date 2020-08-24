# -*- coding: utf-8 -*-
"""
@author: rrizzio@rrizzio.com
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
import time


def entropy(vector, length):
    """
    Calculate the entropy of the distribution of the elements in vector
    
    Parameters
    ----------
    vector : 1D numpy ndarray
        An axis of a numpy ndarray
    length : Integer    
        The number of elements in the vector
        
    Returns
    -------
    Float
        Value of entropy    
    """
    unique, counts = np.unique(vector, return_counts=True)
    S = -1 * np.sum((counts/length) * np.log2(counts/length))
    return S


def exec_time(func):
    """
    Decorator to calculate execution time of a given function

    Parameters
    ----------
    func : function object
        Function whose execution time is to be calculated

    Returns
    -------
    Float
        Execution time
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        Xred, Yred = func(*args, **kwargs)
        etime = time.time() - start
        return Xred, Yred, etime
    return wrapper


def fit_model(X, Y, parameters, dataset):
    """
    Train a Support Vector Classifier using Grid Search cross validation with 5 fold

    Parameters
    ----------
    X : numpy ndarray
        Matrix of predictors
    Y : 1D numpy ndarray
        vector of classes
    parameters : dictionary of lists
        Grid values for the Grid Search algorithm
    dataset : string
        Type of dataset: 'All' or 'Reduced'

    Returns
    -------
    mean_fit_time : numpy array
        Vector with the mean training time for the 5 folds for each combination of the parameters
    grid_search : GridSearchCV object
        Trained model
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
    grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), parameters, cv=5)
    grid_search.fit(X_train, y_train)
    test_score = grid_search.score(X_test, y_test)
    best_parameters = grid_search.best_params_
    mean_fit_time = grid_search.cv_results_['mean_fit_time']
    return mean_fit_time, grid_search, test_score, best_parameters 


@exec_time
def reduce(X, Y, Kneighbors, Kestimate):
    """
    Use KNN to produce reduced X and Y

    Parameters
    ----------
    X : numpy ndarray
        Matrix of predictors
    Y : 1D numpy ndarray
        vector of classes
    Kneighbors : integer
        k to fit k-NN
    Kestimate : integer
        k to estimate from fitted k-NN

    Returns
    -------
    Xred : numpy ndarray
        Reduced Matrix of predictors
    Yred : 1D numpy ndarray
        Reduced vector of classes
    """
    clf = neighbors.KNeighborsClassifier(Kneighbors, weights='distance')
    clf.fit(X, Y)
    Kindices = clf.kneighbors(n_neighbors=Kestimate, return_distance=False)
    Klabels = Y[Kindices]
    length = Klabels.shape[1]
    Svector = np.apply_along_axis(entropy, 1, Klabels, length)
    Svector_filter = Svector > 0
    Xred = X[Svector_filter]
    Yred = Y[Svector_filter]
    return Xred, Yred


    

