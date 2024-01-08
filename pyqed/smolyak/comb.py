#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:51:54 2023

@author: bing
"""

import itertools, operator
import numpy  as np 

def unsorted(balls, boxes):
    rng = list(range(1, balls + 1)) * boxes
    
    return set(i for i in itertools.permutations(rng, boxes) if sum(i) == balls)

balls = 6
boxes = 2

def combinations_with_replacement_counts(n, r):
    size = n + r - 1
    for indices in itertools.combinations(range(size), n-1):
        starts = [0] + [index+1 for index in indices]
        stops = indices + (size,)
        yield tuple(map(operator.sub, stops, starts))

# print(list(combinations_with_replacement_counts(3, 6)))

# def balls_in_boxes(balls, boxes, minimum=1):
#     result = []
#     l = [0 for i in range(0, m)]
#     result.append(put_balls_in_boxes(n, m-1, l, 0, minimum))
#     return result
    
# def put_balls_in_boxes(n, m, l, idx, minimum):
#     if m == 0:
#         l[idx] = n
#         return
    
#     for i in range(minimum, n+1-minimum):
#         l[idx] = i
#         put_balls_in_boxes(n-i, m-1, l, idx+1, minimum)
    
#     return l

def balls_in_boxes(n, m):
    result = []
    l = [0 for i in range(0, m)]
    result.append(put_balls_in_boxes(n, m-1, l, 0))
    return result
    
def put_balls_in_boxes(n, m, l, idx):
    if m == 0:
        l[idx] = n
        return l
    
    for i in range(1, n):
        l[idx] = i
        put_balls_in_boxes(n-i, m-1, l, idx+1)
    
    return l

n = 4
m = 3
result = balls_in_boxes(n, m)
