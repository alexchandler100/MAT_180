# GreogoriOCR
MAT 180 Group Project Proposal
1.	Group Members: Andrew Patton, Cenny Rangel, Kaleb Crans
2.	Project Name: GregoriOCR
3.	Goals:

(a)	GregoBase and Internet Archive will be the databases we use to collect image scans of old antiphonals (our data).

(b)	Using OCR, we want to convert images of antiphonal music (text with neumes) into gabc code.  This code can then be read by Gregorio, which is a LaTeX-like text editor for neume music.  GregoBase is a database where volunteers manually digitize songs from old antiphonals.  Using OCR, we could automate this task.

(c)	Convolutional Neural Networks will be the base of our OCR.

(d)	To measure performance, we’ll write a function to compare our algorithm’s output with the volunteer code from GregoBase.  The output consists of a sequence of plain text and a sequence of gabc code (intermingled).  So, our performance function will compare the sequence of output plain text with the GregoBase plain text.  It will then compare the output gabc and GregoBase gabc and calculate a percentage of accuracy.

![](https://user-images.githubusercontent.com/91860903/204427530-4382e0b3-2f96-4358-a6ac-55709eda9449.png "Fig 2.1 Binary Classifier NN Summary")
