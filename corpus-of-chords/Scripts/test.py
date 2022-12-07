import dataProsessing as dp
import MidiWriting as mw
import os
import json as json
print(os.getcwd())
inv  = json.load(open("Data\\Corpi\\Jsb16thSeparated(t_60_rr_re)\\Jsb16thSeparated(t_60_rr_re)_vocab.json"))["inv_vocab"]
dp.generateMetadata("Data\\Corpi\\Jsb16thSeparated(t_60_rr_re)",inv)