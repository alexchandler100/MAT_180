import dataProsessing as dp
import MidiWriting as mw
import os
import json as json
import generator
import numpy as np
print(os.getcwd())

#this function is for generating new versions of the dataset based on the RAW dataset. Currently rests and repeated chords are
#automaticly removed, and the (musical) key is standardized to middle C. 
#The supported transformation is to add (musical) transpositions to the dataset
#ex. add the dataset transposed up 7 semitones to the dataset
#after generating the transformed dataset the function will write it to disk, than generate the vocab and inverse vocab used in
#many other places in the project. Finally it generated the labled metadata used to power the tensorboard visualiser.

#because I am bad at files, the metadata must manually be copied moved into the corresponding tensorboard data folder
def buildDataset(name, transpositions):
    dataset = json.load(open("Data\\Corpi\\Jsb16thSeparated(RAW).json"))
    dataset = dp.remove_empty_chords(dp.removeRepeatedChords(dataset))
    dataset = sum([dp.standardizeKey(dataset, 60 + i) for i in transpositions],[])
    with open("data\\corpi\\" + name + "\\" + name + ".json", "w") as outfile:
        json.dump(dataset, outfile)
    vocab = dp.generateVocab(dataset)
    with open("data\\corpi\\" + name + "\\" + name + "_vocab.json", "w") as outfile:
        json.dump(vocab, outfile)
    dp.generateMetadata("data\\corpi\\Jsb16thSeparated(t_60+-fifth_rr_re)",vocab["inv_vocab"])


#this function generates the set of the N closes neighbors to each point in the embedding This is later used to generate new sequences
#by starting at some point, than drawing some path through the space. The neighborhood list is than saved to disk because it takes a 
#long time to generate
def generationModel(databaseName, modelName, neighbors):
    embeddings = json.load(open("Data\\Corpi\\" + databaseName + "\\weights\\" + modelName))
    embeddings = [np.array(i) for i in embeddings]
    neighbors = generator.getNeighborhoods(embeddings,10)

    with open("Data\\Corpi\\" + databaseName + "\\weights\\" + modelName + "_neighbors", "w") as outfile:
        json.dump(neighbors, outfile)

#this function uses a generated set of neighborhoods to generate new sequences and saves it to disk
def generateSequence(databaseName,modelName,len,duration = 480,start = 0, name = "test.mid"):
    vocab = json.load(open("Data\\Corpi\\" + databaseName + "\\"+ databaseName + "_vocab.json"))
    inv_vocab = vocab["inv_vocab"]
    neighbors = json.load(open("Data\\Corpi\\" + databaseName + "\\weights\\" + modelName + "_neighbors"))
    sequence = generator.generate(neighbors, start, len)
    sequence = dp.devectorizeSequence(sequence, inv_vocab)
    mw.toMidi(sequence ,duration=duration).save(name)

generateSequence("Jsb16thSeparated(t_60+-fifth_rr_re)", "d_128 n_15", 1000)