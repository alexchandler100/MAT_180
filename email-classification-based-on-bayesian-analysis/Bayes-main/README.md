# Email classification based on Bayesian analysis
Group Members: Yuhuan Ma, Yifei He and Ziqing Tang

### 1. Introduction
Emails have become part of our lives. No matter in school, work place. Most of people will communicate through emails. If you have an email account, we are sure that you have seen emails being categorised into spam and ham. Various email providers employ algorithms to filter emails based on spam and ham. It's wonderful to see machines help us handle the tasks so that we can focus on more important stuff. In this project, we will understand briefly about the Naive Bayes Algorithm and try to label spam email by algorithm. Besides Naive Bayes, other classifier algorithms like Support Vector Machine, or Neural Network also get the job done!

### 2. Data Collection
We will collect datasets from open source. And we try to collect more datasets from Internet. 

*Some online resources for datasets*
*  https://www.kaggle.com/datasets
*  https://archive.ics.uci.edu/ml/datasets.php
*  https://www.csie.ntu.edu.tw/˜cjlin/libsvmtools/datasets/
*  https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research

We find some datasets such as following:

English Spam Dataset
* [Spam Mail 📧 Classifier](https://www.kaggle.com/code/syamkakarla/spam-mail-classifier)
* [smsspamcollection](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/)

Chinese Spam Dataset
* [BayesSpam](https://github.com/shijing888/BayesSpam)


### 3. Method
#### 3.1. Pre-Processing
* i. We remove punctuation, remove tags, remove digits and special chars in email. **re** is a great tool.
* ii. We picked out 80% mails for training and another 20% mails for testing our algorithms and built-in inplementation. **train_test_split** is a great tool provided by the scikit-learn library in Python to finish this task.
* iii. We transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text. **CountVectorizer** is a great tool provided by the scikit-learn library in Python to finish this task.
* iv. Black listing and White listing
#### 3.2. Naive Bayes
 We choose Naive Bayes as learning algorithm used  learning algorithm. Naive Bayes is a probabilistic algorithm based on the Bayes Theorem used for email spam filtering in data analytics. In this situation, the formula can be written as:

![1](https://latex.codecogs.com/svg.image?P(\text{spam}\mid\text{word})=\frac{P(\text{word}\mid\text{spam})P(\text{spam})}{P(\text{word})})

For instance, the probability of the word “FREE” appears in an email is 20%, the probability of an email being a spam is 25%, and the probability of a junk email has the word “FREE” is 45%. Then, when an email contains the word “FREE” was received by a user than the system will calculate the probability of this email is a spam according to the Bayes’ theorem is 56%. At the same time, the cost of classifying a legitimate email into spam is far larger than classifying a junk email into legitimate. So the system might not ignore this email. However, as the amount of data becomes larger, the accuracy will also be improved. According to another study, only when the probability is as high as 99%, they will make the decision and filter this email. 

It further states that

![2](https://latex.codecogs.com/svg.image?P(\text{spam}\mid\text{text})=\frac{P(\text{text}\mid\text{spam})P(\text{spam})}{P(\text{text})}=\frac{P(\text{spam})\prod_{i}P(\text{word}_i\mid\text{spam})}{\prod_{i}P(\text{word}_i)})

![3](https://latex.codecogs.com/svg.image?P(\text{ham}\mid\text{text})=\frac{P(\text{text}\mid\text{ham})P(\text{ham})}{P(\text{text})}=\frac{P(\text{ham})\prod_{i}P(\text{word}_i\mid\text{ham})}{\prod_{i}P(\text{word}_i)})

which assuming that each word occurrence is independent. If ![4](https://latex.codecogs.com/svg.image?P(\text{ham}\mid\text{text})<P(\text{spam}\mid\text{text})), we judged it as a spam email.

We also use build-in implementation including **BernoulliNB(), MultinomialNB() and ComplementNB()** in our project to compare performance, Some information of this built-in implementation can be find here ([naive_bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)).
### 4. Performance
We choose three as indicators to measure your performance of the task, including accuracy, precision [(spam and label as spam)/(labeled as spam)] and recall rate [(spam and label as spam)/(all spam)]. We use **accuracy_score, precision_score, recall_score** in **sklearn.metrics** to calculate them.

Three built-in implementation performace as following in Dataset 1([Spam Mail 📧 Classifier](https://www.kaggle.com/code/syamkakarla/spam-mail-classifier)).
* BernoulliNB():
accuracy: 0.9101, precision: 0.8632, recall: 0.8200
* MultinomialNB():
accuracy: 0.9507, precision: 0.8807, recall: 0.9600
* ComplementNB():
accuracy: 0.9527, precision: 0.8815, recall: 0.9667
* Our methods:
accuracy: 0.9353, precision: 0.8416, recall: 0.9567
### 5. Summary and Outlook
Built-in implementation and our method behave well in spam classification. But it can be improved by some useful pre-processing. For example, Porter Stemmer algorithm can be considered in project, The purpose of stemming is to bring variant forms of a word together, not to map a word onto its 'paradigm' form. It can decrease the demension of vector on the basis of the frequency (count) of each word. Then, we only consider word whether or not appear in text in our work. We can take frequency (count) of each word into consideration. Many other indicators, like the domain type of the sender (.edu or .org), or whether it has an attachment or not, should also be taken into consideration in real email spam filtering. We should be carefully to choose features for  email spam filtering.
### 6. How to Use this Project
All our tasks are in **'spam_classified.ipynb'**. You need to run each cell by steps. You only need to run one of cells in 1.2.1. and 1.2.2 to choose dataset. If you want to use other datasets, you should organize your dataset to include at least **'text'** and **'label_num'** as in **'spam_ham_dataset.csv'**. If not, as 1.2.2. DataSet2, you should add 'text' and 'label_num' by yourself. Finishing training data, you can paste your email content in 5. Output will tell if it is a spam email or not.
