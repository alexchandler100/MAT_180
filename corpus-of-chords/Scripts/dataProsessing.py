import io
import music21
import numpy as np
import json
import os

#---------------------------------File_Retreival----------------


def loadDataset(name):
    dataset = json.load(open(name))
    newdataset = tuplefy(dataset)
   
    return newdataset

def saveDataset(data, name):
    with open("Data\\" +  name, "w") as outfile:
        json.dump(data, outfile)

#--------------------PREPROCESSING---------------------------

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

def identifyChord(chord):
    chord = [note for note in chord if note != -1]
    return music21.chord.Chord(chord).pitchedCommonName()
    
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

def remove_empty_chords(sequence):
    retseq = []
    for chord in sequence:
        if chord != [-1,-1,-1,-1]:
            retseq.append(chord)
    return retseq

#----------------------------------------------------------------------    

#-----------------PROCESSING-------------------------------------------
def generateVocab(dataset):
    vocab,inv_vocab, i = {},[],0
    for sequence in dataset:
        for chord in sequence:
            if chord not in inv_vocab:
                vocab[str(chord)] = i
                inv_vocab.append(chord)
                i += 1
    print(i)
    print(len(inv_vocab))
    return vocab, inv_vocab


def vectorizeDataset(dataset,vocab):
    vectorizedDataset = []
    for sequence in dataset:
        vectorizedDataset.append([vocab[chord] for chord in sequence])
    return vectorizedDataset

#-----------------------------TENSORBOARD_OUTPUT---------------------------

def generateMetadata(dir, inv_vocab):
    out_m = io.open(os.path.join(dir, 'metadata.tsv'), 'w', encoding='utf-8')

    for i in range(len(inv_vocab.items())):
        out_m.write(str(inv_vocab[i]) + identifyChord(i) + "\n")

    out_m.close()
