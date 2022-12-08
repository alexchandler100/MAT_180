#scripts regarding running gradient descent

import numpy as np
from numpy import linalg as LA
from scripts.polynomial import add_poly_terms 

def J(X,y,v,lambda_):
    ######################### your code goes here ########################
    rowx = len(X)
    J = LA.norm(X@v-y)**2
    re = LA.norm(v,2)**2
    return (1/rowx)*J + lambda_*re

def DJ(X,y,v,lambda_):
    ######################### your code goes here ########################
    rowx = len(X)
    gradient = X.T@X@v-X.T@y
    reg = 2*v
    return (2/rowx)*gradient + lambda_*reg

def GD_linreg_improved(X,y,epsilon,lambda_,max_iters = 10000): 
    ######################### your code goes here ########################
    a = X.shape
    v = np.zeros([a[1],1])
    H = (2/a[0])*X.T@X+2*lambda_*np.identity(a[1])
    costs = [J(X,y,v,lambda_)]
    for i in range(max_iters):
        alpha = (DJ(X,y,v,lambda_).T@DJ(X,y,v,lambda_))/(DJ(X,y,v,lambda_).T@H@DJ(X,y,v,lambda_))
        v = v-DJ(X,y,v,lambda_)*alpha
        costs.append(J(X,y,v,lambda_))
        if i % 1000 == 0:
            print(f'After {i} steps the cost is {costs[i]}')
        if abs(costs[i] - costs[i-1])<epsilon:
            break
    print(f'After {i} steps the cost is {costs[i]}')
    return v,costs

def fit(X, y, epsilon, lambda_, max_iters = 10000, poly_terms = 1):
    print(f'Running polynomial regression of degree {poly_terms} \n')
    
    v, costs =  GD_linreg_improved(add_poly_terms(X, poly_terms), y, epsilon, lambda_, max_iters) 
    
    print(f'\nFinal cost is {costs[-1]}\n')
    return v, costs