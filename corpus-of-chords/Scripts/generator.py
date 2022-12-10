import numpy as np

def getNeighborhoods(vectors,n):
    neighborhoods = {}
    n = n + 1
    for i,v in enumerate(vectors):
        
        if i % 100 == 0: 
            print(i)
        d = []
        for w in vectors:
            d.append(np.linalg.norm(v - w))
        p = np.argpartition(d,n)
        p = [int(p[j]) for j in range(n) if p[j] != i]
        neighborhoods[i] = p
    return neighborhoods

def generate(neighborhoods, start,len):
    current = start
    sequence = [start]
    for i in range(len):
        
        next = np.random.choice(neighborhoods[str(current)])
        sequence.append(next)
        current = next
    return sequence