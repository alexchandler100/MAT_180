import dataProsessing as dp
import MidiWriting as mw
import os
import json as json
print(os.getcwd())
corpus = dp.loadDataset("Jsb16thSeparated.json")
newcorpus = {}
newcorpus["train"] = dp.standardizeKey(corpus["train"],60)
newcorpus["test"] = dp.standardizeKey(corpus["test"],60)
newcorpus["valid"] = dp.standardizeKey(corpus["valid"],60)

newcorpus["train"] = list(map(dp.removeRepeatedChords,newcorpus["train"]))
newcorpus["test"] = list(map(dp.removeRepeatedChords,newcorpus["test"]))
newcorpus["valid"] = list(map(dp.removeRepeatedChords,newcorpus["valid"]))
with open("Data\Jsb16thSeparated(t_60_rr).json", "w") as outfile:
    json.dump(newcorpus, outfile)