import random

def getRandomWord(n, S):
    res = []
    for i in range(n):
        res.append(S[random.randint(0, len(S) - 1)])
    return res

def free_part(kh):
    res = {}
    for key1 in kh.keys():
        for key2 in kh[key1].keys():
            a = kh[key1][key2]
            gens = a.gens()
            n = len([gen for gen in gens if gen.additive_order() == +Infinity])
            res[(key1, key2)] = n
    return res

def torsion_part(kh): 
    res = {}
    for key1 in kh.keys():
        for key2 in kh[key1].keys():
            val = {}
            a = kh[key1][key2]
            gens = a.gens()
            for gen in gens:
                b = gen.additive_order()
                if b == +Infinity:
                    continue
                elif b in val.keys():
                    val[b] += 1
                else:
                    val[b] = 1   
            if len(list(val.keys()))>0:
                res[(key1, key2)] = val
    return res

def count_FP(a_FP):
    return sum(a_FP[key] for key in a_FP.keys())

def count_TP(a_TP):
    #return sum([a_TP[key]][2] for key in a_TP.keys())
    return sum(a_TP[key][2] for key in a_TP.keys())