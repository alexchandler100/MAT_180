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

#identifychord:
    #chord: vector with midi numbers
    #moreinfo: i
            #f false, returns chord name
            #if true, returns [chord name, chord root, chord quality]
def identifyChord(chord,moreinfo = False):
    if tuple(chord) == (-1,-1,-1,-1):
        if not moreinfo:
            return ("Empty Chord")  
        else:
            return np.array(["Empty Chord","Empty Chord","Empty Chord"])
    chord = [note for note in chord if note != -1]
    if not moreinfo:
        return music21.chord.Chord(chord).pitchedCommonName
    chord21 = music21.chord.Chord(chord)
    return np.array([chord21.pitchedCommonName, chord21.root().name, chord21.quality])

#getChordLabels:
    #chords: list of chords in midi vector form
    #moreinfo: determines output content per identifyChord
    #verbose: if true, prints step # every 300 steps
def getChordLabels(chords,moreinfo=False,verbose=False):
    labels = [] 
    for i in range(len(chords)):
        if verbose and (i%300 == 0):
            print(f'{i}th step')
        labels.append(np.array(identifyChord(chords[i],moreinfo)))
    return np.array(labels)
    
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
        
        vectorizedDataset.append([vocab[str(chord)] for chord in sequence])
    return vectorizedDataset

#-----------------------------TENSORBOARD_OUTPUT---------------------------

def generateMetadata(dir, inv_vocab):
    out_m = io.open(os.path.join(dir, 'metadata.tsv'), 'w', encoding='utf-8')

    for i in inv_vocab:
    
        out_m.write(str(i) + identifyChord(i) + "\n")

    out_m.close()
    
#---------------------------PROCESSING FOR CLUSTER TABLEAUS----------------
def getFreqs(indices, data):
    datacount = coll.Counter([data[i] for i in indices])
    total = sum(datacount.values())
    freq = [[data[i],round(datacount[data[i]]/total,4)] for i in indices]
    return freq

def mostcommon(n,freq):
    order = np.argsort(np.array(freq)[:,1])
    this = [freq[o] for o in order[len(order)-n:len(order)]]
    return this

def partitionFreqs(partition,labels):
    namefreqs = []
    rootfreqs = []
    qualfreqs = []
    partsizes = []
    for p in range(k):
        p_indices = [i for i in range(len(partition)) if partition[i]==p]
        namefreqs += [getFreqs(p_indices,labels_km[:,0])]
        rootfreqs += [getFreqs(p_indices,labels_km[:,1])]
        qualfreqs += [getFreqs(p_indices,labels_km[:,2])]
        partsizes += [round(len(p_indices)/len(weights),4)]
    return (namefreqs,rootfreqs,qualfreqs,partsizes)

def tabulate_partitions(feature,freqs,n,k):
    title = f'Partitions, {feature}'
    tabledata = []
    for p in range(k):
        tabledata += [[p] + [partsizes[p]] + mostcommon(n,freqs[p])]
    headers = ['Partition','part. size']
    for i in range(n): #TODO: format
        headers += [f'{i+1} most common']
    return (title,tabledata,headers)
