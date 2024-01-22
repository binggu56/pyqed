#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 01:34:00 2024

neurol network problem fit APES

NN model from 
https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/

@author: bingg
"""

import numpy as np
import torch

def init_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        params['b'+str(l)] = np.zeros((layer_dims[l], 1))
        
    return params

def sigmoid(Z):
	A = 1/(1+np.exp(np.dot(-1, Z)))
    cache = (Z)
    
    return A, cache

class NeuralNetwork:
    pass
