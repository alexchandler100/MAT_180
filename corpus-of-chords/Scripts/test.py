import dataProsessing as dp
import MidiWriting as mw
import os
import json as json
print(os.getcwd())
corpus = dp.loadDataset("data\\Corpi\\Jsb16thSeparated(t_60_rr_re)\\Jsb16thSeparated(t_60_rr_re).json")

vocab = dp.generateVocab(corpus)
vocab = {"vocab" : vocab[0], "inv_vocab":vocab[1] }

dp.saveDataset(data = vocab, name = "Jsb16thSeparated(t_60_rr_re)_vocab.json")