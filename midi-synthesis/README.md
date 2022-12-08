
# MIDI Synthesis with Feedforward Neural Network

## Group Members
Jada Yip, Harkins Loh, Timothy Ng

## Table of Contents
 
- [Preface](#preface)
- [Our Project](#our-project)
	- [MIDI File Preprocessing](#midi-file-preprocessing)
		- [Overview on .mid file formatting](#overview-on-mid-file-formatting)
		- [Preprocessing the things](#preprocessing-the-things)
		- [Creating the datasets for the neural network](#creating-the-datasets-for-the-neural-network)
	- [Training the Neural Network](#training-the-neural-network)
  - [Generating Music](#generating-music)
  - [Results](#results)
 - [Methods](#methods)



## Preface

Many music today are written in 12-tone equal temperament, with more than a single instrument (multiple tracks) and 
synchronized by time (tempo). The developers of
the [MIDI](https://www.midi.org) protocol have developed the .mid file to store, modify and load
musical data of this nature in a digital medium which encompasses
a good deal of music that can be systematically read. 
Thanks to its convenient design,
the .mid file format is used in many digital audio
workstations (DAW), electronic instruments and score editors today.

Like many digital pieces of data, its ability to interact with non-human
systems raise the perennial empirical question *vis-a-vis*
music: can computers generate good-sounding musical pieces?

To this answer, we are obligated philosophically to respond in the negative.
Outside of philosophy however, and constraining our music to be short piano pieces,
 we demonstrate that it can, in a sense. Our neural network and generation
 algorithm reliably allows us to generate short musical pieces in a given 
 genre with passable accuracy.


## Our Project
Our dataset consists of around 11,000 MIDI files from the
[ADL Piano Midi](https://github.com/lucasnfe/adl-piano-midi). Every
MIDI file in this dataset has to be processed in order to be used by our
neural network. The sections below will give a brief overview of how our
data is organized, and how you can train and generate your own samples.

### MIDI File Preprocessing
#### Overview on .mid file formatting
Every .mid file contains several `tracks`. In each of these `tracks`, there are
`messages`. The first track contains `messages`
of the .mid file in terms of musical structure, such as key signature, tempo and
time signature of the *entire* music, so a key signature and time signature were
to change at a later time, it is reflected in a `message` in this `track`.

The second and successive tracks contain data of note events. Every note is
an object with three important features: `pitch`, `velocity` and `time`. These are
all integers. On the 12-tone equal temperament scale, the corresponding
pitch to the integer used by midi can be found [here](https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies).
`velocity` is a measure of loudness for each note, and a value of 80 corresponds to *f* (forte)
in musical notation. Notes which have `velocity = 0` are instructions to turn off that note.
`time` is a measure, in ticks, of how long to wait before this note is read. So for
example, to play a C4 (middle C) at forte for 200 ticks, the track messages would look like
`[('note_on', note =  60, velocity = 80, time = 0),('note_on', note =  60, velocity = 0, time = 200)]`.


#### Preprocessing the things
We will use [Mido](https://mido.readthedocs.io/en/latest/index.html) to interface between
Python and .mid files. The ADL dataset, when unzipped, is a folder
containing main genres, and in each of these genre folder are additional
folders detailing the subgenre, and then another folder by author and finally the .mid file
itself. Thus, from the path directory `\Data`, the .mid file is at least 3 folders deep.

We have developed a function `collect_midi_files` to extract all .mid files in
`\Data`, so you can add your own .mid file into the directory `\Data` anywhere without worrying about
folder structure (make sure you add it into a folder with a genre for supervised learning! So for example, you should add Liszt's pieces to `\Data\Classical`). The same function will return a dictionary `genres_dict` which
contains as keys the music genre, and each key has a list of names of .mid files
associated to that genre. So an example `genres_dict` will look like `{'Ambient': ['Sunburst (Album Version).mid',...]}`.

However, some files from the ADL dataset are regrettably corrupted. We have
developed countermeasures to remove corrupted data from computation (without deleting
the source file). This can be done by invoking the `cleanData()` function, but
this will take about an hour on the current dataset. 

To this end, we have uploaded a `uncorrupted.txt` file which can be loaded
in the notebook into the `genres_dict` that already have corrupted files removed.
If you are adding your own .mid files, make sure they are not corrupted, or you can
use the `cleanUp()` method to remove them.

#### Creating the datasets for the neural network
Because we are using a feedforward neural network, the number of notes must match
the architecture of the input layer. The nodes of the input layer can be specified
by changing the constant `CAPACITY` (default is 256, which is equivalent to 16 bars in common time).
The `compress` method given in the notebook 
will pad .mid files with insufficient notes, and remove notes beyond `CAPACITY`.

With `genres_dict` defined, we can generate a parent dataset `X` and labels `y`
via the methods under the section **Formatting the dataset** in the notebook.
This will split `X` and `y` into 9 datasets to be trained; training, validation and test sets for 
`X_pitches`, `X_velocities` and `X_times` respectively. These methods will also automatically
pad every entry of the dataset as described in the previous paragraph.

### Training the Neural Network
As alluded to multiple times, this project uses a feedforward neural network. Popular
activation functions such as `ReLU`, `Sigmoid`, `Softmax`, `Squared` and `Linear` have been provided,
with instructions on how to add your own in the relevant sections in the notebook. Backpropagation
methods are also given, so you will only need to worry about getting the right architecture for
your own model.

Under the section **Fitting it all** in the notebook, you can specify the
desired architecture, activations and parameters to train your datasets. In
the cells which follow after it, you can plot costs computed by the network,
and save your results to .json files so you can reload high-performing models.

The motivation behind three networks is the idea that music is beyond just pitches;
the velocities and time deltas are also important to convey musical information, and
will be paramount to our music generation algorithm. We view pitch as the most responsible for melodic lines,
velocity as the mood, and time deltas as the rhythm.

## Generating Music
This is the *raison-d'etre* of the project, and the algorithm can be found at the bottom of the notebook along with useful information on how it works.
After training the neural network to obtain high-performing models, the `generate` method will return a .mid file
to the length of `CAPACITY`. You can specify what genre it should generate, and
even give an incomplete .mid file for it to finish (as long as it is within `CAPACITY`).

Futhermore, you can specify the note, velocity and time delta ranges for the generator.
The algorithm considers the entire output holistically; on every new note to be added, it re-reads the
entire file for accuracy changes and selects the note with the best accuracy given these conditions. It considers
pitch accuracy, velocity accuracy and time delta accuracy based on the model given by the neural networks.

Hence, the algorithm performs best when the Pitch Neural Network, Velocity Neural Network and Time Deltas Neural Network returns
a model that has high accuracy on the test set. Our pitch model has some merit, although the other models require further parameter tuning
and architecture reconsideration.

## Results
Our model consists of three neural networks, each one responsible for fitting pitch, velocity and
time deltas respectively.  

Our neural network for pitch is 79% accurate on its training set with `CAPACITY = 256` and `removeZeroVelocities = False`, but in validation and
test it is consistently 40%. We have provided the architecture, weights, biases and activations for this model in the "Pitch NN" folder, and
can be loaded in the notebook with the `load("Pitch NN")` method.

The other neural networks for fitting velocities and time deltas were less successful;
accuracies on validation sets were less than 10% for both after days of tweaking parameters and selecting architectures, and this represents a need for
further study and refinement on the compatibility of MIDI data to be used in feedforward neural networks.

Notwithstanding the latter two results, we can proceed with generating melodic lines alone with 
the generation algorithm. An example output with our current model can be found in the Output folder.

## Methods
This section documents all of the custom methods we defined in the project. Many of these
will be highly useful for preprocessing.

**getNotes(mid,trackNo)**  
given a MidiFile `mid`,
returns a list of Messages in `trackNo` of `mid` that have the `note_on` attribute.

**getNotesTrack(track)**  
given a MidiTrack `track`, returns a list of Message in `track` that have the `note_on` attribute.

**quantize16(notes)**  
given list of `Messages` with the `note_on` attribute, returns a list of notes such that
the fastest notes are quantized to a semiquaver (16th note).

**absolutize(notes)**  
given list of `Messages` with the `note_on` attribute, returns a list of notes
such that every `time` attribute is the exact time when it is played. That is, all `time` attributes
are no longer deltas; they now refer to time distances from the beginning of the music.

**trim(notes,capacity, pad = True, removeZeroVelocities = True)**  
given a list `notes`, integer `capacity`, boolean values for `pad` and `removeZeroVelocities`,
returns a list of notes such that its length is within `capacity`, and if the original
list was under `capacity` already, loop the original list of notes until it reaches `capacity`.
If `removeZeroVelocities` is true, this method only returns notes with nonzero velocity attributes.

**getMidiFile(name)**  
given string `name`, ending with .mid, this method will search and return a MidiFile
of that `name` in `\Data`.

**compress(mid, pad = True, removeZeroVelocities = True)**  
this is the main method used to turn .mid files into the correct form to be used by the
neural networks. In many MIDI, there are usually more than one instrument track being played at
the same time. This method compresses all of them into a single track. Given MidiFile `mid`, `pad` and `removeZeroVelocities` options,
returns a compressed MidiFile.

**decoder(array)**  
given a one-hot encoding of genres stored as a numpy array `array`, returns the name of that
genre corresponding to those in `genres_dict`.

**encoder(genre)**  
given a string `genre`, returns a one-hot encoding of that genre corresponding to those in `genres_dict`.

**genADLData(size)**  
given integer `size`, returns two outputs: the first output is a list of
genres, randomly sampled, of size `size`, and the second output is a list of string names of .mid files
associated with the genres list in the first output.

**compressListofNames(genre,music)**  
given list of strings `genre` and list of strings `music`, compresses every entry in
`music` (equivalent to applying the compress method earlier elementwise), and additionally
checks if the files can be compressed, modifying `genre` to account for deleted files.
This returns a list of compressed entries in `music`.

**cleanData()**  
this removes entries of `genres_dict` which are corrupted.
