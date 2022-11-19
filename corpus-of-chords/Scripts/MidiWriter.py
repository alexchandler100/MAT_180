import mido
from DataTypes import *

def compositionToMidiFile(composition:Composition):
    midiFile = mido.MidiFile()
    midiFile.ticks_per_beat = composition.metaData.ticksPerBeat
    if composition.metaData.timeSignature != ():
        midiFile.tracks.append(generateTempoMap(composition.metaData))
    for track in composition.tracks:
        midiFile.tracks.append(generateMidiTrack(track))
    return midiFile

def generateTempoMap(timeData):
    tempoMap = mido.MidiTrack()
    timeSignature = mido.MetaMessage(type = "time_signature", numerator = timeData.timeSignature[0],  denominator = timeData.timeSignature[1],
    clocks_per_click = timeData.timeSignature[2],  notated_32nd_notes_per_beat = timeData.timeSignature[3])
    tempo = mido.MetaMessage(type = "set_tempo", tempo = timeData.tempo)
    tempoMap.extend([timeSignature,tempo])
    return tempoMap

def generateMidiTrack(track):
    midiTrack = mido.MidiTrack()
    midiTrack.append(mido.MetaMessage(type = "track_name", name = track.ID))
    emptyEvent = Event(startTime = 0, duration = 0, noteSet = NoteSet(frozenset(),frozenset()))
    lastEvent = emptyEvent
    for nextEvent in track.events:
        midiTrack.extend(generateMidiMessages(lastEvent, nextEvent))
        lastEvent = nextEvent
    midiTrack.extend(generateMidiMessages(lastEvent, emptyEvent))
    return midiTrack



def generateMidiMessages(lastEvent, nextEvent):
    assert lastEvent != nextEvent
    messages = []
    #for every note that was in the last event but is not in the next one generate a note off
    for note in lastEvent.noteSet.allNotes() - nextEvent.noteSet.allNotes():
        messages.append(mido.Message(type = "note_off", note = note.value, channel = note.channel, velocity = 0, time = 0))
    
    #for every not that was in the last event but is being resounded in the next, generate a note off, resounding done in next step
    #note:this step may be unnessisary but I think it generates cleaner files
    for note in lastEvent.noteSet.allNotes() & nextEvent.noteSet.sounded:
        messages.append(mido.Message(type = "note_off", note = note.value, channel = note.channel, velocity = 0, time = 0))

    #for every note that is just sounded in the next event generate a note on
    for note in nextEvent.noteSet.sounded:
        messages.append(mido.Message(type = "note_on", note = note.value, channel = note.channel, velocity = note.velocity, time = 0))

    #update the first events delta to the last events duration
    if len(messages) == 0:
        pass
        #print(lastEvent)
        #print(nextEvent)
    if len(messages) > 0:
        messages[0].time = lastEvent.duration
    return messages