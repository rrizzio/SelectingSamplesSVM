# -*- coding: utf-8 -*-
"""
@author: rrizzio@rrizzio.com
"""

import numpy as np
from utils import fit_model, reduce
from visualization import make_plots, make_score_heatmap, make_time_boxplot

# Parameters
instances = 50000
Kneighbors = 5
Kestimate = 5
param1 = 'C'
param2 = 'gamma'
grid1 = [0.001, 0.01, 0.1, 1, 10, 100]
grid2 = [0.001, 0.01, 0.1, 1, 10, 100]

# Create the hyperparameters grid
parameters = {param1: grid1, param2: grid2}

# Create a synthetic dataset with XOR classes
np.random.seed(0)
X = np.random.randn(instances, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# Fit the model
all_mean_fit_time, all_grid_search, test_score, best_parameters = fit_model(X, Y, parameters, 'All')
print("All - Best Test Score:", test_score)
print("All - Best Parameters:", best_parameters)

# Create datapoints plots
make_plots(X, Y, all_grid_search, 'All')

# Create scores heatmap
make_score_heatmap(parameters, param1, param2, all_grid_search, 'All')

# Get the reduced dataset 
Xred, Yred, rtime = reduce(X, Y, Kneighbors, Kestimate)

# Fit the model on the reduced dataset
reduced_mean_fit_time, reduced_grid_search, test_score, best_parameters = fit_model(Xred, Yred, parameters, 'Reduced')
print("Reduced - Best Test Score:", test_score)
print("Reduced - Best Parameters:", best_parameters)
print("Reduce Time(sec):", rtime)

# Create datapoints plots
make_plots(Xred, Yred, reduced_grid_search, 'Reduced')

# Create scores heatmap
make_score_heatmap(parameters, param1, param2, reduced_grid_search, 'Reduced')

# Create boxplots of the training times
adj_reduced_mean_fit_time = reduced_mean_fit_time + rtime
make_time_boxplot(all_mean_fit_time, adj_reduced_mean_fit_time)