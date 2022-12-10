# GreogoriOCR
MAT 180 Group Project Proposal
1.	Group Members: Andrew Patton, Cenny Rangel, Kaleb Crans
2.	Project Name: GregoriOCR
3.	Goals:

(a)	GregoBase and Internet Archive will be the databases we use to collect image scans of old antiphonals (our data).

(b)	Using OCR, we want to convert images of antiphonal music (text with neumes) into gabc code.  This code can then be read by Gregorio, which is a LaTeX-like text editor for neume music.  GregoBase is a database where volunteers manually digitize songs from old antiphonals.  The goal of using OCR is to hopefully automate this task.

(c)	Convolutional Neural Networks will be the base of our OCR.

(d)	To measure performance, we’ll write a function to compare our algorithm’s output with the volunteer code from GregoBase.  The output consists of a sequence of plain text and a sequence of gabc code (intermingled).  So, our performance function will compare the sequence of output plain text with the GregoBase plain text.  It will then compare the output gabc and GregoBase gabc and calculate a percentage of accuracy.

We aimed high, but the target was very far.  What we ended up with is the starting point of an Optical Music Recognition (OMR) application.  As will be discussed later, we came upon many roadblocks in our journey.  We ended up creating two models.  Our first is a convolutional neural net that scans a music staff and counts the number of notes in the image.  Our second model takes an image of a single note and uses a CNN to classify the pitch of the note.  (WE ALSO HAVE LABEL IPYNB)

How well it performs:

Correct note count: Image.  Our total data set is only 30 elements and our test set is 3, so our accuracy is usually 0%, but is sometimes 33% if we’re lucky.

Pitch Classifier: 20%

Instructions: 

CNN_count.ipynb:

1.	Go to Gregobase: https://gregobase.selapa.net/scores.php.

2.	Download a gabc file that is about one staff long as well as the corresponding png (image showing this)

Note: The gabc code cannot span more than one staff.

3.	The image is then added to the “Chant_Data” folder

4.	The number of notes in the gabc file must be manually counted.  The number is then appended to the “Chant data label.txt” file

5.	CNN_count.ipynb uses as the X dataset “Chant_Data” and “Chant data label.txt” as the y labels.  Theoretically, it would predict the number of notes in whichever images it uses as test data.

pitch_classifier.ipynb:

1.	Go to GregoBase and download an image of a Gregorian chant

2.	Crop out a single note as a 138 by 34 png.

3.	Add that image to the “Note Data” folder

4.	Append the pitch of that note to “Note_label.txt”, e.g., if the pitch is g, append (g) to the file.

5.	pitch_classifier.ipynb uses this data to predict the pitch of the notes in “Note Data”. 

Roadblocks and Solutions

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

![](https://github.com/CennyBo/GregoriOCR/blob/74c5c454b0fe2f0b47335f722b38f34eb506b042/gregori-ocr/Gregorio%20Read%20Me%20Images/Three%20Note.png)

(yes the smiley face is necessary as it is latin text).

And so on.  As presented, this becomes incredibly more complex the more chants are added to the dataset.   Here is a complicated score for example:

![](https://github.com/CennyBo/GregoriOCR/blob/74c5c454b0fe2f0b47335f722b38f34eb506b042/gregori-ocr/Gregorio%20Read%20Me%20Images/Example%20Score.png)

And the associated label: 

(c3) THis(gxfe/gvFE) is(fd) the(fh) day(hhhvFEfef)

Where in the parenthesis are the notes and outside are the plain text. 

Another large issue was the dataset itself. The dataset is labeled however it is labeled with a Text file which poses a few issues. The text must be scrapped for the notes in the chants which is a simple enough task, but for most of the labeled data there are multiple pdfs that must be stitched together by hand as the way the data is formatted within the gregorio database is not built to handle large data processing such as this: 

Solution: Hire someone else to stitch together pdfs, or build a separate project to skim large datasets and find connections.

This issue is similar to issue one but stands on its own to get its own place, and the issue is normalizing the scores from the raw pages. There must be bounding boxes around the score as the note identifying algorithm only needs to see this part of the raw data. Most of the raw data (the breviaries themselves) are not scanned but instead are photographs and because of this some of the scores can be tilted, distorted, and in some cases illegible to most. 

Solution: Write a CNN to give a 1 if it is a scanned, grayscale document where the scores are straight. And a 0 if it is a photograph. If a 0 is outputted it would need to be processed to intelligently undistort the images, put them into grayscale without losing too much information, and then fed back into to CNN to see whether it needs more processing or if it can be read by the bounding box algorithm
Bad Data:

![](https://github.com/CennyBo/GregoriOCR/blob/74c5c454b0fe2f0b47335f722b38f34eb506b042/gregori-ocr/Gregorio%20Read%20Me%20Images/Bad%20Raw%20Data.png)

Good Data:

![](https://github.com/CennyBo/GregoriOCR/blob/74c5c454b0fe2f0b47335f722b38f34eb506b042/gregori-ocr/Gregorio%20Read%20Me%20Images/Good%20Raw%20Data.png)
