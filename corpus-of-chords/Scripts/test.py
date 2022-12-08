import dataProsessing as dp
import MidiWriting as mw
import os
import json as json
import generator
import numpy as np
print(os.getcwd())


#dataset = json.load(open("Data\\Corpi\\Jsb16thSeparated(RAW).json"))
#dataset = dp.remove_empty_chords(dp.removeRepeatedChords(dataset))
#dataset = sum([dp.standardizeKey(dataset, 60 + i) for i in [0,5,7]],[])
#with open("data\\corpi\\Jsb16thSeparated(t_60+-fifth_rr_re)\\Jsb16thSeparated(t_60+-fifth_rr_re).json", "w") as outfile:
    #json.dump(dataset, outfile)
#vocab = dp.generateVocab(dataset)
#with open("data\\corpi\\Jsb16thSeparated(t_60+-fifth_rr_re)\\Jsb16thSeparated(t_60+-fifth_rr_re)_vocab.json", "w") as outfile:
    #json.dump(vocab, outfile)
#dp.generateMetadata("data\\corpi\\Jsb16thSeparated(t_60+-fifth_rr_re)",vocab["inv_vocab"])

#embeddings = json.load(open("Data\\Corpi\\Jsb16thSeparated(t_60+-fifth_rr_re)\\weights\\d_128 n_15"))
#embeddings = [np.array(i) for i in embeddings]
#neighbors = generator.getNeighborhoods(embeddings,10)

#with open("Data\\Corpi\\Jsb16thSeparated(t_60+-fifth_rr_re)\\weights\\d_128 n_15", "w") as outfile:
    #json.dump(neighbors, outfile)

vocab = json.load(open("Data\\Corpi\\Jsb16thSeparated(t_60+-fifth_rr_re)\\Jsb16thSeparated(t_60+-fifth_rr_re)_vocab.json"))
inv_vocab = vocab["inv_vocab"]
neighbors = json.load(open("Data\\Corpi\\Jsb16thSeparated(t_60+-fifth_rr_re)\\weights\\d_128 n_15_neighbors"))
sequence = generator.generate(neighbors, 0, 100)
sequence = dp.devectorizeSequence(sequence, inv_vocab)
mw.toMidi(sequence ,duration=480, verbose = True).save("test.mid")