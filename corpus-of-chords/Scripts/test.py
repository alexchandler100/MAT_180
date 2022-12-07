import dataProsessing as dp
import MidiWriting as mw
import os
import json as json
import generator
import numpy as np
print(os.getcwd())


vocab = json.load(open("Data\\Corpi\\Jsb16thSeparated(t_60_rr_re)\\Jsb16thSeparated(t_60_rr_re)_vocab.json"))
inv_vocab = vocab["inv_vocab"]
neighbors = json.load(open("Data\\Corpi\\Jsb16thSeparated(t_60_rr_re)\\weights\\d_128 n_15_neighbors"))
 


sequence = generator.generate(neighbors, 0, 50)
print(sequence)
translatedsequence = []
for chord in sequence:
    translatedsequence.append(inv_vocab[chord])
print(translatedsequence)
mw.toMidi(translatedsequence,duration=480).save("test.mid")