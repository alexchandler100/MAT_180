import mido
import numpy as np
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