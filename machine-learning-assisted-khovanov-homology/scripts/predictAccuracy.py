import numpy as np
def prediction(x,v):
    return round((x@v)[0])

def accuracy(X,v,y):
    count = 0
    for i,x in enumerate(X):
        if prediction(x,v) == y[i][0]:
            count += 1
    return count / len(X)
