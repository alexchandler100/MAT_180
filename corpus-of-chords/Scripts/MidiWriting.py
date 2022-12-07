import mido
import numpy as np
def toMidi(sequence, duration):
    def seqeunceToMidiTrack(sequence,legato= "true") :



        messages = mido.MidiTrack()
        rest = 0
        if not legato:
            for note in sequence:
                if(note == -1):
                    rest = rest+duration
                else:
                    messages.append(mido.Message(type = "note_on", note = note, channel = 0, velocity = 127, time = 0))
                    messages.append(mido.Message(type = "note_off", note = note, channel = 0, velocity = 127, time = duration + rest))
                    rest = 0

        elif legato:
            
            previousNote = -2
            
            for note in sequence:
                print("note is " ,note)
                print("previous note is ", previousNote)
                if(previousNote == -2):
                    print("initializing " ,note)
                    messages.append(mido.Message(type = "note_on", note = note, channel = 0, velocity = 127, time = 0))
                    rest += duration
                elif previousNote == note:
                    print("holding " ,note)
                    rest += duration
                elif note == -1 and previousNote != -1:
                    print("ending " ,previousNote , " and resting")
                    messages.append(mido.Message(type = "note_off", note = previousNote, channel = 0, velocity = 127, time = rest))
                    rest = 0
                else:
                    print("ending " ,previousNote , " and starting " , note)
                    messages.append(mido.Message(type = "note_off", note = previousNote, channel = 0, velocity = 127, time = rest))
                    messages.append(mido.Message(type = "note_on", note = note, channel = 0, velocity = 127, time = 0))
                    rest = 0
                previousNote = note

        return messages

    midiFile = mido.MidiFile()
    splitSequences = np.array(sequence).T.tolist()
    for sequence in splitSequences:
        midiFile.tracks.append(seqeunceToMidiTrack(sequence))
    return midiFile