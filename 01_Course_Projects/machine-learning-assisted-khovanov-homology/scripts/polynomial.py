# script for adding polynomial terms to an m by n matrix of data points.

import numpy as np

def generate_monomials_eq(n,k):
    if n == 1: 
        yield [k,]
    else:
        for i in range(k+1):
            for j in generate_monomials_eq(n-1,k-i):
                yield [i,] + j

def generate_monomials_leq(n,k): 
    L = []
    for i in range(k+1):
        Lh = list(generate_monomials_eq(n,i))
        Lh.sort(reverse = True)
        L += Lh
    return L

def add_poly_terms(X,k):
    n = len(X[0])
    mons = generate_monomials_leq(n,k)
    L = []
    for mon in mons:
        X1 = np.ones(len(X))
        for i in range(n):
            X1 *= X[:,i]**mon[i]
        L.append(X1)
    return(np.array(L).T)
