from typing import List, Dict
import copy
from util import colorStructureString as colorString
from colorama import Fore

class Note:
    def __init__(self, value:int, velocity:int, channel:int):
        self.value = value
        self.velocity = velocity
        self.channel = channel
        
    def transpose(self, semitones):
        
        return Note(self.value + semitones, self.velocity, self.channel)
    def setVelocity(self, velocity):
        
        return Note(self.value, velocity, self.channel)

    def __hash__(self):
        return hash(self.value) ^ hash(self.velocity) ^ hash(self.channel) 

    def __eq__(self, other):
        return (
            self.value == other.value and
            self.velocity == other.velocity and
            self.channel == other.channel
            
        )

    def __str__(self):
        return colorString("Note", [
                ("value", self.value),
                ("velocity", self.velocity),
                ("channel", self.channel)
            ],
            nameColor=Fore.LIGHTMAGENTA_EX
        )

    def __repr__(self):
        return self.__str__()



class NoteSet:

    def __init__(self, sustained, sounded):
        self.sustained = frozenset(sustained)
        self.sounded = frozenset(sounded)

    def __eq__(self, other):
        return self.sustained == other.sustained and self.sounded == other.sounded

    def __hash__(self):
        return hash(self.sustained) ^ hash(self.sounded)

    def allNotes(self):
        return self.sustained | self.sounded
    
    def transpose(self, semitones):
        return NoteSet(frozenset(map(lambda s : s.transpose(semitones), self.sustained)), frozenset(map(lambda s : s.transpose(semitones), self.sounded)))
    
    def setVelocities(self, velocity):
        return NoteSet(frozenset(map(lambda s : s.setVelocity(velocity), self.sustained)), frozenset(map(lambda s : s.setVelocity(velocity), self.sounded)))
    
    def __str__(self):
        return colorString("noteSet",[
            ("sustained", self.sustained),
            ("sounded" , self.sounded)
        ],
            nameColor = Fore.LIGHTMAGENTA_EX, valueNewLine= True, iterNewLine= False)
    def __repr__(self):
        return self.__str__()


class MetaData:
    def __init__(self, ticksPerBeat):
        self.timeSignature = ()
        self.tempo = 0
        self.ticksPerBeat = ticksPerBeat


        
    
    def __str__(self):
        return colorString("MetaData", [
            ("timeSignature", self.timeSignature),
            ("tempo", self.tempo),
            ("ticksPerBeat", self.ticksPerBeat)
        ])
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return (
            self.timeSignature == other.timeSignature and
            self.tempo == other.tempo and
            self.ticksPerBeat == other.ticksPerBeat
        )
 
class Event:
    def __init__(self, startTime: int, duration: int, noteSet):
        self.startTime = startTime
        self.measureStartTime = 0
        self.duration = duration
        self.noteSet = noteSet

    def __eq__(self, other):
        return (
            self.startTime == other.startTime and
            self.measureStartTime == other.measureStartTime and
            self.duration == other.duration and
            self.noteSet == other.noteSet
        )
    
    def __str__(self):
        from colorama import Fore
        return colorString("Event", [
                ("startTime", self.startTime),
                ("duration", self.duration),
                ("noteSet", self.noteSet),
            ],
            nameColor=Fore.LIGHTBLUE_EX,
            memberColor=Fore.LIGHTRED_EX,
            valueColor=Fore.YELLOW
        )
    
    def __repr__(self):
        return self.__str__()

class Composition:
    def __init__(self, name, metaData = None):
        self.metaData = metaData
        self.name = name
        self.tracks = []
    
    def __str__(self):
        retString = "Composition: {}:\nMeta: {}\n".format(self.name, self.metaData)
        for track in self.tracks :
            retString += str(track) + "\n"
        return colorString("Composition",[
                ("name", self.name),
                ("meta", self.metaData),
                ("tracks", self.tracks),
            ]
        )
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        print("composition compairisons currently not compairing name")
        return (
            self.metaData == other.metaData and
            frozenset(self.tracks) == frozenset(other.tracks)
        )
   
    def standardizeTrackNames(self):
        for i, track in enumerate(self.tracks):
            track.name = "track " + str(i + 1)
    def toCompositeEventList(self):
        composites = CompositeEventList(self.metaData, self.name)
        tracks = copy.deepcopy(self.tracks) 
        

        def minDuration():
            durations = []
            for track in tracks:
                if len(track.events) > 0:
                    durations.append(track.events[0].duration)
            return min(durations)
        
        def getPartialComposite(duration, track):
           
            if len(track.events) == 0:
                return (track.ID, NoteSet(frozenset(), frozenset()))
            
            
            returnSet = (track.ID , copy.copy(track.events[0].noteSet))

            track.events[0].duration -= duration
            if track.events[0].duration == 0:
                track.events.pop(0)
                return returnSet

            
            track.events[0].noteSet.sustained = track.events[0].noteSet.allNotes()
            track.events[0].noteSet.sounded = frozenset()
            
            return returnSet

        def tracksNotEmpty():
            for track in tracks:
                if len(track.events) > 0:
                    return True
            return False
        
        while tracksNotEmpty():
            nextCompositeEventTrackSet = set()
            nextCompositeEventDuration = minDuration()
            for track in tracks:
                nextCompositeEventTrackSet.add(getPartialComposite(nextCompositeEventDuration, track))
            composites.addCompositeEvent(CompositeEvent(nextCompositeEventTrackSet, nextCompositeEventDuration))
        return composites


class Track:
    def __init__(self, ID, metaData, events:list = None):
        self.ID = ID
        self.metaData = metaData
        self.events = [] if events == None else events
    
    def __str__(self):
        return colorString("Track", [
                ("Track id", self.ID),
                ("meta data", self.metaData),
                ("events", self.events)
            ],
            valueNewLine=True,
            memberColor=Fore.LIGHTMAGENTA_EX,
            nameColor=Fore.BLUE
        )

    def __hash__(self):
        return hash(self.ID)
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.ID == other.ID and self.metaData == other.metaData and self.events == other.events
  

class CompositeEvent:
    def __init__(self, tracks, duration):
        #tracks should be a fozen set of tuples where the first element is a track ID and the second element is a noteSet
        #todo(not urgent) care about measure start time in any way shape or form
        self.tracks = frozenset(tracks)
        self.duration = duration
    
    def getTrackIDs(self):
        return {track[0] for track in self.tracks}
    
    def getNoteSet(self, ID):
        for track in self.tracks:
            if track[0] == ID:
                return track[1]
        return NoteSet(frozenset(), frozenset())
    
    def setVelocities(self, velocity):
        return CompositeEvent(frozenset(map(lambda s : (s[0], s[1].setVelocities(velocity)), self.tracks)), self.duration)
        
    def transpose(self, semitones):
        return CompositeEvent(frozenset(map(lambda s : (s[0], s[1].transpose(semitones)), self.tracks)), self.duration)
    def __eq__(self, other):
        return self.tracks == other.tracks and self.duration == other.duration
    
    def __hash__(self):
        return hash(self.tracks) ^ hash(self.duration)

    def __str__(self):
        return "CompositeEvent({}, {})".format(self.duration, {x[0]:x[1] for x in self.tracks})
    
    def __repr__(self):
        return self.__str__()

class CompositeEventList:

    def __init__(self, timeData, name):
        self.eventList = []
        self.trackIDs = set()
        self.timeData = timeData
        self.name = name

    def addCompositeEvent(self, event:CompositeEvent):
        self.trackIDs.update(event.getTrackIDs())
        self.eventList.append(event)
    
    def __str__(self):
        return "CompositeEventList({})".format(self.eventList)
    
    def __repr__(self):
        return self.__str__()
    #-------------------------TRANSFORMATIONS-------------------------#
    def setDurations(self, durations):
        for event in self.eventList:
            event.duration = durations
    def removeSilence(self):
        
        for event in copy.copy(self.eventList):
            for track in event.tracks:
                empty = True
                if len(track[1].allNotes()) > 0:
                    empty = False
            if(empty == True):
                self.eventList.remove(event)
    
    def transpose(self, semitones):
        returnEventList = copy.deepcopy(self)
        if semitones == 0:
            return returnEventList
        returnEventList.name = returnEventList.name + " transpose " + str(semitones)

        for i, event in enumerate(returnEventList.eventList):
            returnEventList.eventList[i] = event.transpose(semitones)

        
        return returnEventList

    def setVelocities(self, velocity):
        returnEventList = copy.deepcopy(self)
        returnEventList.name = returnEventList.name + " velocity: " + str(velocity)

        for i, event in enumerate(returnEventList.eventList):
            returnEventList.eventList[i] = event.setVelocities(velocity)

        
        return returnEventList

    def generateRhythemMap(self):
        returnEventList = []

        for event in self.eventList:
            returnEventList.append(event.duration)      
        return returnEventList

    def applyRhythemMap(self, rhythemMap):
        for i, event in enumerate(self.eventList):
            event.duration = rhythemMap[i % len(rhythemMap)]
            
                


    #-------------------------TRANSFORMATIONS-------------------------#

    

    def toComposition(self):
        def checkExtention(lastNoteSet, nextNoteSet):
            return lastNoteSet.allNotes() == nextNoteSet.allNotes() and len(nextNoteSet.sounded) == 0
        
        composition = Composition(name = self.name, metaData = self.timeData)
        globalTime = 0
        for trackID in self.trackIDs:
            composition.tracks.append(Track(trackID, self.timeData))


        for event in self.eventList:
            for track in composition.tracks:
              
                if len(track.events) == 0 or not checkExtention(track.events[-1].noteSet, event.getNoteSet(track.ID)):

                    track.events.append(Event(startTime = globalTime, duration = event.duration, noteSet = event.getNoteSet(track.ID)))
                else: 
                    track.events[-1].duration += event.duration
                
            globalTime += event.duration

        return composition
            #find corresponding ID in the composition




