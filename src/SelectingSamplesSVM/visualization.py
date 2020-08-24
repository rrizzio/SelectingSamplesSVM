# -*- coding: utf-8 -*-
"""
@author: rrizzio@rrizzio.com
"""

import numpy as np
import matplotlib.pyplot as plt
from heatmap import heatmap, annotate_heatmap

def make_plots(X, Y, grid_model, dataset):
    """
    Produce three scatter plots:
        - X,Y scatter plot
        - X,Y scatter plot with decision hyperplane
        - X,Y scatter plot with decision hyperplane and support vectors

    Parameters
    ----------
    X : numpy ndarray
        Matrix of predictors
    Y : 1D numpy ndarray
        vector of classes
        DESCRIPTION.
    grid_model : GridSearchCV object
        Trained model        
    dataset : string
        Type of dataset: 'All' or 'Reduced'

    Returns
    -------
    Produce chart files in directory ../../target/visualization
    """
    svc = grid_model.best_estimator_
    sv = svc.support_vectors_
    # defining the dimensions and the grid for plotting
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot the datapoints on the grid
    plt.figure() 
    plt.scatter(X[:, 0], X[:, 1], s=10, c=Y, cmap=plt.cm.Paired, edgecolors='k')
    title = dataset + ' Datapoints for Two XOR Classes'
    plt.title(title)
    outfile = '../../target/visualization/' + title + '.png'
    plt.savefig(outfile, format="png")
    # plot the datapoints on the grid with the decision boundaries
    plt.figure() 
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='solid', colors='red')
    plt.scatter(X[:, 0], X[:, 1], s=10, c=Y, cmap=plt.cm.Paired, edgecolors='k')
    title = dataset + ' Datapoints Decision Contours'
    plt.title(title)
    outfile = '../../target/visualization/' + title + '.png'
    plt.savefig(outfile, format="png")    
    # plot support vectors
    plt.figure()
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles='solid', colors='red')
    plt.scatter(X[:, 0], X[:, 1], s=10, c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.scatter(sv[:, 0], sv[:, 1], s=10, facecolors='none', zorder=10, linewidth=3, color='green')
    title = dataset + ' Datapoints Support Vectors'
    plt.title(title)
    outfile = '../../target/visualization/' + title + '.png'
    plt.savefig(outfile, format="png")
    # plt.show()
    
def make_score_heatmap(parameters, param1, param2, grid_model, dataset):
    """
    Produce heatmap of mean scores for each combination of the parameters

    Parameters
    ----------
    parameters : dictionary of lists
        Grid values for the Grid Search algorithm
    param1 : string
        The parameter C
    param2 : string
        The parameter gamma
    grid_model :  GridSearchCV object
        Trained model
    dataset : string
        Type of dataset: 'All' or 'Reduced'

    Returns
    -------
    Produce chart files in directory ../../target/visualization
    """
    scores = [score for score in grid_model.cv_results_['mean_test_score']]
    scores = np.array(scores).reshape(6, 6)
    fig, ax = plt.subplots()
    im, cbar = heatmap(scores, parameters[param1], parameters[param2], ax=ax,
                       cmap="YlGn", cbarlabel="Mean Scores")
    annotate_heatmap(im, valfmt="{x:.2f}")
    ax.set_xlabel(param2)
    ax.set_ylabel(param1)
    title = dataset + ' Datapoints GridSearchCV Mean Scores'
    ax.set_title(title)
    fig.tight_layout()
    outfile = '../../target/visualization/' + title + '.png'
    plt.savefig(outfile, format="png")
    # plt.show()
    
def make_time_boxplot(all_mean_fit_time, adj_reduced_mean_fit_time):
    """
    Produce Boxplots of the mean fit time for All and Reduced datasets

    Parameters
    ----------
    all_mean_fit_time : numpy ndarray
        Mean fit time for all folds of cross validation for all combination 
        of the parameters for dataset All
    adj_reduced_mean_fit_time : numpy ndarray
        Mean fit time for all folds of cross validation for all combination 
        of the parameters for dataset Reduced

    Returns
    -------
    Produce chart files in directory ../target/visualization
    """
    data = [all_mean_fit_time, adj_reduced_mean_fit_time]
    labels = ['all_mean_fit_time', 'adj_reduced_mean_fit_time']
    fig1, ax1 = plt.subplots()
    title = 'Cross Validation Mean Fit Time'
    ax1.set_title(title)
    ax1.boxplot(data, notch=False, vert=True, patch_artist=True, labels=labels) 
    ax1.yaxis.grid(True)
    ax1.set_ylabel('Seconds')
    outfile = '../../target/visualization/' + title + '.png'
    plt.savefig(outfile, format="png")
    # plt.show()    