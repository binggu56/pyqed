#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 00:06:31 2024

@author: bingg
"""

def balls_in_boxes(n, m, minimum=1):
    l = [0 for i in range(0, m)]
    result = []
    put_balls_in_boxes(n, m-1, l, 0, result, minimum)
    return result
    
def put_balls_in_boxes(n, m, l, idx, result, minimum):
    if m == 0:
        l[idx] = n
        result.append(l.copy())
        # print(l)
        return
    
    for i in range(minimum, n-minimum+1):
        l[idx] = i
        put_balls_in_boxes(n-i, m-1, l, idx+1, result, minimum)
        
n = 5
m = 3
result = balls_in_boxes(n, m)
print(result)