
from io import StringIO
import music21
import numpy as np
import json

def loadDataset(name):
    dataset = json.load(open("Data\\" + name))
    newdataset = {}
    newdataset["train"] = tuplefy(dataset["train"])
    newdataset["valid"] = tuplefy(dataset["valid"])
    newdataset["test"] = tuplefy(dataset["test"])
    return newdataset

def tuplefy(dataset):
    newDataset = []
    for sequence in dataset:
        newDataset.append(list(map(tuple,sequence)))
    return newDataset
        
def getKey(sequence):
    return music21.chord.Chord(sequence[0]).root().midi

def transpose(sequence, semitones):
    f = lambda note : note if note == -1 else note + semitones
    f = np.vectorize(f)
    return f(np.array(sequence)).tolist()
    
def standardizeKey(sequences, key):
    retsequences = []
    for sequence in sequences:
        retsequences.append(transpose(sequence, key - getKey(sequence)))
    return retsequences

def removeRepeatedChords(sequence):
    previousChord = []
    retSequence = []
    for i in range(len(sequence)):
        if sequence[i] != previousChord:
            retSequence.append(sequence[i])
        previousChord = sequence[i]
    return retSequence

    