#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:01:40 2024

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import string
# from pyqed import 
alphabet = list(string.ascii_lowercase)
D = 2

ini = "".join(alphabet[:D]) + 'x' + "".join(alphabet[D:2*D])+'y, ' + \
    "".join(alphabet[D:2*D])+'y -> ' + "".join(alphabet[:D]) + 'x'




print(ini)