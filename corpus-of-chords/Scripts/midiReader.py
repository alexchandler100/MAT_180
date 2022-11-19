from DataTypes import *
import mido
from copy import deepcopy

def defaultTrackGenerator(unprocessedTracks:list):
    for track in unprocessedTracks:
        globalTime = 0
        sustainedNotes = set()
        soundedNotes = set()
        events = []
        name = track[0]

        for i, message in enumerate(track[1]):

            if message.time != 0:
                events.append(Event(
                    globalTime, 
                    message.time,
                    NoteSet(deepcopy(sustainedNotes), deepcopy(soundedNotes))
                ))
                sustainedNotes.update(soundedNotes)
                soundedNotes.clear()
                globalTime += message.time

            if message.type == "note_off": 
                for note in (sustainedNotes | soundedNotes):
                    if (note.value == message.note and note.channel == message.channel):
                        sustainedNotes.discard(note)
                        soundedNotes.discard(note)
                        break

            elif message.type == "note_on": #add new notes
                noteAlreadyOn = False
                for note in sustainedNotes | soundedNotes:
                    if (note.value == message.note and note.channel == message.channel):
                        note.velocity = message.velocity
                        noteAlreadyOn = True
                        if note not in soundedNotes:
                            sustainedNotes.discard(note)
                            soundedNotes.add(note) 
                        break
                if not noteAlreadyOn:
                    soundedNotes.add(Note(
                        value = message.note,
                        channel = message.channel,
                        velocity = message.velocity)
                    )

        yield Track(name, None, events)

def defaultMetadataHandler(midiFile:mido.MidiFile):
    #TODO REWRITE THIS WHOLE FUNCTION WITH MIDI 1 STANDARDS IN MIND
    #strip the file of metadata and return MIDI only tracklist coupled with the event list with metadata attached

    composition = Composition(midiFile.filename) #the end list of note events
    composition.metaData = MetaData(midiFile.ticks_per_beat)

    midiTrackList = [] #the tracks in the midi file
    
    for track in midiFile.tracks:
        midiTrackCount = 0
        isMidi = False
        metadata = [] #todo: do something with track meta data
        messages = []
        trackName = ''
        for message in track[:]:
            if message.is_meta:

                if message.type == 'time_signature':
                    composition.metaData.timeSignature = (
                        message.numerator, 
                        message.denominator, 
                        message.clocks_per_click, 
                        message.notated_32nd_notes_per_beat
                    )
                elif message.type == 'set_tempo':
                    composition.metaData.tempo = message.tempo
                elif message.type == 'track_name':
                    trackName = message.name

            elif message.type == 'note_on' or message.type == 'note_off':
                isMidi = True
                messages.append(message)

        if isMidi:
            midiTrackCount += 1
            if trackName == '':
                trackName = "untitledTrack " + str(midiTrackCount)
            midiTrackList.append((trackName, messages))
    
    return (composition, midiTrackList)
        
def buildComposition(file:mido.MidiFile, eventGenerator = defaultTrackGenerator, metaDataHandler = defaultMetadataHandler):
    #make a composition from the file's meta data and build a list of unproccessed tracks
    composition, midiTrackList = metaDataHandler(file)

    for track in eventGenerator(midiTrackList):
        track.metaData = composition.metaData
        composition.tracks.append(track)

    return composition