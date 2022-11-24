import json as json
from io import StringIO
import music21
import mido
import numpy as np

corpus = json.load(open("Jsb16thSeparated.json"))

def toMidi(sequence, duration):
    def seqeunceToMidiTrack(sequence):
        messages = mido.MidiTrack()
        rest = 0
        for note in sequence:
            if(note == -1):
                rest = rest+duration
            else:
                messages.append(mido.Message(type = "note_on", note = note, channel = 0, velocity = 127, time = 0))
                messages.append(mido.Message(type = "note_off", note = note, channel = 0, velocity = 127, time = duration + rest))
                rest = 0
        return messages

    midiFile = mido.MidiFile()
    splitSequences = np.array(sequence).T.tolist()
    for sequence in splitSequences:
        midiFile.tracks.append(seqeunceToMidiTrack(sequence,duration))
    return midiFile




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



newcorpus = {}
newcorpus["train"] = standardizeKey(corpus["train"],60)
newcorpus["valid"] = standardizeKey(corpus["valid"],60)
newcorpus["test"] = standardizeKey(corpus["test"],60)
with open("jsb16thSeperated(t_60).json", "w") as outfile:
    json.dump(newcorpus, outfile)

