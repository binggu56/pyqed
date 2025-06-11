#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 23:57:19 2024

@author: bingg

Refs

"""
import itertools

def combination(n, m):
    """
    Calculate the combination :math:`C_{n}^{m}`,

    .. math::

        C_{n}^{m} = \\frac{n!}{m!(n-m)!}.

    Parameters
    ----------
    n: int
       Number n.
    m: int
        Number m.

    Returns
    -------
    res: int
        The calculated result.

    Examples
    --------
    >>> import edrixs
    >>> edrixs.combination(6, 2)
    15

    """

    if m > n or n < 0 or m < 0:
        print("wrong number in combination")
        return
    if m == 0 or n == m:
        return 1

    largest = max(m, n - m)
    smallest = min(m, n - m)
    numer = 1.0
    for i in range(largest + 1, n + 1):
        numer *= i

    denom = 1.0
    for i in range(1, smallest + 1):
        denom *= i

    res = int(numer / denom)
    return res

def fock_bin(n, k):
    """
    Return all the possible :math:`n`-length binary
    where :math:`k` of :math:`n` digitals are set to 1.

    Parameters
    ----------
    n: int
        Binary length :math:`n`.
    k: int
        How many digitals are set to be 1.

    Returns
    -------
    res: list of int-lists
        A list of list containing the binary digitals.

    Examples
    --------
    >>> import edrixs
    >>> edrixs.fock_bin(4, 2)
    [[1, 1, 0, 0],
     [1, 0, 1, 0],
     [1, 0, 0, 1],
     [0, 1, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 1]]

    """

    if n == 0:
        return [[0]]

    res = []
    for bits in itertools.combinations(list(range(n)), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        res.append(s)
    return res

def basis(n, m):
    pass

print(fock_bin(4, 2))