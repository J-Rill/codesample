#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
By: Joachim Rillo

This is a coding sample which uses gradient descent to estimate the locations 
of 9 cities in the USA without information about their latitudes and longitudes,
using their relative distances.

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

#1ii

# Defining the distance Matrix

D_matrix = np.array([[0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949], 
        [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
        [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
        [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
        [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996], 
        [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
        [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
        [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
        [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])

# List of cities
labels = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']

D = pd.DataFrame(D_matrix, columns = labels, index = labels)

# In the function below I will define the big X as a NumPy array 
# that contains all possible coordinates for each state:

# Defining the discrepancy function-- we want to minimize the distance between
# the initial predicted guess and the actual guess.

def discrepancy_fn(X, D_m):
    n = D_m.shape[0]
    sum_disc = 0
    
    for i in range(1, n):
        x_i = X[i]
        for j in range(i+1, n): #Since D is a symmetric matrix
            x_j = X[j]
            D_ij = D_m.iloc[i,j]
            dist = np.linalg.norm(x_i-x_j)
            
            ssr = (dist-D_ij)**2
            sum_disc += ssr
    
    return sum_disc

# Defining the derivative of this discrepancy function above
    
def discrepancy_prime(X, x_i, i, D_m):
    n = D_m.shape[0]
    prime_sum = 0
    for j in range(1, n):
        if i != j:
            x_j = X[j]
            dist = np.linalg.norm(x_i-x_j)
            city_i = labels[i]
            city_j = labels[j]
            D_ij = D_m[city_i][city_j]
            
            # Using what we calculated in part a
            term1 = 2*(dist-D_ij) 
            term2 = (x_i-x_j)/dist
            prime_sum += term1*term2
    
    return prime_sum

# Running the gradient descent algorithm

def gradient_descent(X, D_m, alpha, max_iter):
    n = D_m.shape[0]
    epsilon = 15
    disc = math.inf
    
    for i in range(max_iter):
        
        new_disc = discrepancy_fn(X, D_m) # update discrepancy function
        
        # If the difference between the updated guess and the actual distance
        # is small, we move on to the next pair of cities.
        
        if disc - new_disc  < epsilon: 
            break
        
        disc = new_disc
        
        X_fix = X.reshape(n, 2) 
        
        for j in range(n):
            x_i = X_fix[j]
            grad = discrepancy_prime(X, x_i, j, D_m)
            norm = np.linalg.norm(grad)
            
            if norm > 0:
                grad = grad / norm
                
            X[j] = X[j] - alpha * grad
        
            
    return X

# Initial random guess for X:

X0 = np.random.rand(9, 2)

# Run gradient descent

X = gradient_descent(X0, D, 3, 100000)

X_fix = X.reshape((-1,2))

# Extract latitude and longitude from optimized X
latitudes = X_fix[:, 0]
longitudes = X_fix[:, 1]

# Plot the cities on a map
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(latitudes, longitudes, color='blue', s = 100)

# Add labels to each city
for i, label in enumerate(labels):
    ax.annotate(label, (latitudes[i], longitudes[i]))

# Set axis labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Relative distance between cities')

plt.show()