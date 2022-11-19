import json as j
from io import StringIO
import music21
import mido
corpus = j.load(open("Jsb16thSeparated.json"))

def toMidi(sequence):
 




def getKey(sequence):
    return music21.chord.Chord(sequence[0]).root().midi


def standardizeKey(sequences, key):
    pass



print(getKey(corpus["train"][0]))