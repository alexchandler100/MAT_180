# GreogoriOCR
MAT 180 Group Project Proposal
1.	Group Members: Andrew Patton, Cenny Rangel, Kaleb Crans
2.	Project Name: GregoriOCR
3.	Goals:

(a)	GregoBase and Internet Archive will be the databases we use to collect image scans of old antiphonals (our data).

(b)	Using OCR, we want to convert images of antiphonal music (text with neumes) into gabc code.  This code can then be read by Gregorio, which is a LaTeX-like text editor for neume music.  GregoBase is a database where volunteers manually digitize songs from old antiphonals.  Using OCR, we could automate this task.

(c)	Convolutional Neural Networks will be the base of our OCR.

(d)	To measure performance, we’ll write a function to compare our algorithm’s output with the volunteer code from GregoBase.  The output consists of a sequence of plain text and a sequence of gabc code (intermingled).  So, our performance function will compare the sequence of output plain text with the GregoBase plain text.  It will then compare the output gabc and GregoBase gabc and calculate a percentage of accuracy.

Final Write-up:

What it accomplishes:

This project is the starting point for an Optical Music Recognition (OMR) application.  The ultimate goal would be to convert an entire page of Gregorian chant into a gabc file.  We did not do that.  That turned out to be a gargantuan task, which was impossible to finish in time.  (More details: What would that entail?)

Our revised approach set out to accomplish the foundational requirements.  Our system takes a png of a single music staff as input, and then counts the number of notes in that image.  It also performs OMR on individual notes.

How well it performs:

Correct note count: ?%

Individual Note OMR Accuracy: ?%

Walk through how a new user can use your model on their own data after cloning the repository:

By following instructions in your README.md file, I should easily be able to run your algorithm on a new example not in your dataset and see that the model performs up to your claimed performance measure.

Instructions: Go to Gregobase: https://gregobase.selapa.net/scores.php.

Download a gabc file that is about one staff long as well as the corresponding png (image showing this)

Note: The gabc code must be should be at least the length of a full staff

The image is then used as input in our image in our encoding function?

The program will output the number of notes in the staff

Future Problems and Solutions
This is a very barebones project and more or less a proof of concept for a more complicated project that would be a serious undertaking, that would exceed the time limit and available information currently present. Some issues we have currently encountered that have hindered progress for this project have been:

Creating bounding boxes for the notes after identifying the amount of notes per score. This is currently unobtainable as there is no labeled dataset for identifying each note besides by hand. 

Solution: with enough time and soul-sucking effort one could manually prepare bounding boxes around each note to feed to a CNN to identify notes. However, this leads into the next issue which arose:

The complexity of GABC and gregorian chants. The characters present within the chants are incredibly complex and vary by many factors. The GABC code is very straightforward and allows one to transcribe notes with knowledge of the chants. However,  this is an issue for our neural network as for single notes our network performs well in identification, but for longer and more complex strings, such as double notes, the order of the notes written affects the output and exponentially increases the outputs our CNN must possess. 

Example of what our CNN would have to classify: 
![](https://github.com/CennyBo/GregoriOCR/blob/74c5c454b0fe2f0b47335f722b38f34eb506b042/gregori-ocr/Gregorio%20Read%20Me%20Images/Example%20notes.png)

Solution: Have multiple CNN’s to identify three things: 

1). Is this a double note or single note? Triple? Quadruple? 

2). This is *blank* type of note! What keys are needed to create this note? e-f? e-h?

3). These *blank* make up the multi-note! What order must the notes be in to create the correct note? Are there any non alphabetical characters needed?

Here is an example of such (note there are alphabetical 13 characters including abc):
Image goes here: (Three Note)
(yes the smiley face is necessary as it is latin text).

And so on.  As presented, this becomes incredibly more complex the more chants are added to the dataset.   Here is a complicated score for example:
Image goes here: (Example Score)
And the associated label: 

(c3) THis(gxfe/gvFE) is(fd) the(fh) day(hhhvFEfef)

Where in the parenthesis are the notes and outside are the plain text. 

Another large issue was the dataset itself. The dataset is labeled however it is labeled with a Text file which poses a few issues. The text must be scrapped for the notes in the chants which is a simple enough task, but for most of the labeled data there are multiple pdfs that must be stitched together by hand as the way the data is formatted within the gregorio database is not built to handle large data processing such as this: 

Solution: Hire someone else to stitch together pdfs, or build a separate project to skim large datasets and find connections.

This issue is similar to issue one but stands on its own to get its own place, and the issue is normalizing the scores from the raw pages. There must be bounding boxes around the score as the note identifying algorithm only needs to see this part of the raw data. Most of the raw data (the breviaries themselves) are not scanned but instead are photographs and because of this some of the scores can be tilted, distorted, and in some cases illegible to most. 

Solution: Write a CNN to give a 1 if it is a scanned, grayscale document where the scores are straight. And a 0 if it is a photograph. If a 0 is outputted it would need to be processed to intelligently undistort the images, put them into grayscale without losing too much information, and then fed back into to CNN to see whether it needs more processing or if it can be read by the bounding box algorithm
Bad Data:
Image goes here: (Bad Raw Data.png)
Good Data:
Image goes here: (Good Raw Data.png)
