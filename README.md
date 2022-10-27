# SentimentAnalysis
Developed using MATLAB, this program classifies given tweets into the sentiment it thinks it represents best. 

# Defining the task
This problem involves the automatic classification of sentiment from recorded tweets from Twitter. Our task is to study the dataset, 
prepare it for machine learning, and to select the best classification model for automatically determining the sentiment displayed by the tweet. 
This is a classification task and will require a supervised learning approach.

# Dataset
The dataset for this problem is contained in the [SentimentAnalysisFata.zip](https://github.com/badroben/SentimentAnalysis/blob/main/SentimentAnalysisData.zip) file. This zip file contains 1 csv file: “text_emotion_data.csv”. 
The csv file contains two columns:
1. “sentiment”: this column lists one of four sentiments (“relief”, “happiness”, “surprise”, “enthusiasm”) that have been assigned to the corresponding tweet.
2. “Content”: this column contains the text of the tweet.<br/>


There are 8651 entries in this file.

# Data Preperation 
In order to use the tweets and their labels, we imported the csv file into MATLAB as a table, built a Bag of Words containing all of the tokenised tweets, 
removed stop words, removed any words with fewer than 100 occurrences in the bag, and built the full Term Frequency-Inverse Document Frequency matrix (tf-idf) for the resulting bag.
We also built a corresponding label vector from the column of sentiments. 

# Features and Lables
We created one feature matrix for training by selecting the first 6921 rows of the tf-idf matrix and all columns. We also created a corresponding label vector with the first 6921 labels.
We created one feature matrix for testing by selecting all rows of the tf-idf matrix after row 6921 (i.e. the remaining rows). We also created a corresponding label vector.

# Model training 
We trained and compared three classification algorithms implemented in MATLAB ([K-Nearest Neighbour algorithm](https://www.ibm.com/uk-en/topics/knn), [Naive Bayes algorithm](https://www.kdnuggets.com/2020/06/naive-bayes-algorithm-everything.html), and [Decision Tree algorithm](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)). 

# Evaluation
Accuracy is measured by comparing the predictions from your models to the test labels in the dataset. It will be sufficient to calculate the accuracy as 
__𝑎𝑐𝑐𝑢𝑟𝑎𝑐𝑦 = 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑐𝑜𝑟𝑟𝑒𝑐𝑡 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠𝑡𝑜𝑡𝑎𝑙 divided by 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑙𝑎𝑏𝑒𝑙𝑠__. 

We also investigated the results more thoroughly by analysing the resulting confusion matrix. We obtained this with the confusionchart function.

# Final Report
__The content of the data set:__ <br/>
The data set available is a CSV file containing textual data obtained from tweets varying in length from long ones to short 2 to 3-word ones, 
each tweet demonstrates a certain feeling from the four sentiments: relief, happiness, surprise, and enthusiasm. The dataset contains a total of 8651 rows, 
and every tweet is labelled with a corresponding sentiment and there are no missing values. 
However, when analysing the data I noticed that some tweets can reflect many sentiments, for example, one tweet says “Finally off work” and it is labelled as happiness, 
thinking about it a person can also be relieved when getting off work and finishing their daily tasks; not sure what standard was used to label the data that relate to more 
than one sentiment, but it is not a black or white situation, it has some ambiguity within and some tweets can be really hard to label, also, some tweets have almost 
the same content but labelled with different sentiments, for instance, a tweet says “Good Morning Sunshines!” is labelled as happiness while another tweet says “good morning sunshine!” 
is labelled as enthusiasm. Moreover, tweets may contain unformal language phrases such as acronyms. For instance, words like asap (as soon as possible) or lol (laugh out loud) are used frequently.<br/>
__The data preparation method:__<br/>
The steps taken to prepare the data for the model are simple. First, I loaded the data into MATLAB as a table and I separated the two columns and transformed them from a table into an array, 
one for the labels (sentiments) and the other one is for the tweets (sentences). After that, I tokenized the sentences to construct a document containing all the words and punctuations in the tweets as tokens. 
Furthermore, I created the bag of words based on the tokenised tweets contained in the document, removed stop words since they do not help in finding the context or the true sentiment of the sentence, 
and removed words with fewer than 100 occurrences. Lastly, I built the tf-idf for the resulting bag to help us evaluate how relevant and important each of those words are to a document in a collection of documents. 
I ended up with a sparse double matrix which I converted to a full matrix to use for training and testing purposes. This step is crucial in any machine learning model especially since we are dealing with textual data, 
so we have to convert it into a form that the computer understands. I have not deviated from the recommended steps since they are the standard protocol to prepare data for any model, 
however, I have tried a few ideas to see if they would affect the result or not, for instance, I tried to compare the results with the existence of punctuation in the bag of words and without it and the result 
was that without the punctuation the accuracy dropped about 1% which is not much but we can conclude that punctuation can help in determining a sentiment. 
The data split was 80/20, I noticed when increasing this, no major changes were recorded from the accuracies.<br/>
__The model training and evaluation methods used:__<br/>
I have tried the K-Nearest Neighbour model first since it is the simplest form of machine learning algorithms for classification. KNN classifies new points based on the distance between that point and its k neighbours. 
The second model I implemented was the Naïve Bayes model, which is a simple classification model that utilises Bayes’ probability theorem to classify data points. The last model I used was the Decision Tree model, 
which classifies new data by continuously splitting it according to a parameter.
The process used to invoke these methods was to prepare the data first through a series of steps that lead to a full matrix representing the tf-idf of words with more than 100 occurrences, 
this was split into an 80/20 ratio with 80% of the data to be used for training and the rest for testing. The training features which resulted from the tf-idf were invoked into the methods fitcknn(), fitcnb(), and fitctree() which resulted in ClassificationKNN model, ClassificationNaiseBayes model, and ClassificationTree model respectively.
For the KNN model, it originally had k = 1, so I attempted to change the value of the number of nearest neighbours to 2, 3, and then 4 which led to an increase in accuracy every time I increase k, reaching about 50% accuracy when k = 20. 
I have also tried changing the distance metric however, some of the metrics were not appropriate to use in this project such as correlation (reason: some points have small relative standard deviations) and cosine (reason: some points have small relative magnitudes)
while the other metrics did not affect the accuracy that much.
For the Naïve Bayes model, I tried to change the parameters to see whether it will enhance the result in some way, but I couldn’t do it since the properties are read-only.
For the Decision Tree model, I tried to optimise hyperparameters like the cross-validation loss of the classifier by adding “OptimizeHyperparameters’ and ‘auto’ to the method parameters, this increased the accuracy to about 51%. <br/>
__The selected model and criteria:__<br/>
Looking at the accuracies of the models used, we can see that the KNN model had an accuracy of 37.67%, the Naïve Bayes one had 28.42%, while the Decision Tree model had
43.96% accuracy. These accuracies are liable to change depending on the parameters specified, as we have seen the KNN model’s accuracy had increased by changing the number k, 
so did the Decision Tree model after optimising hyperparameters.
From the confusion matrices of all 3 models shown below: <br/>
![image](https://user-images.githubusercontent.com/60741379/198392877-6ea93dad-d4c6-499b-8841-be6c1c8a87a9.png) <br/>
![image](https://user-images.githubusercontent.com/60741379/198392938-ee44f138-7701-4dde-956e-d39556d266a5.png) <br/>
![image](https://user-images.githubusercontent.com/60741379/198392987-a6b9a488-12e5-405e-ad38-9d550a84ab67.png) <br/>
we can see that most of the correct predictions were for happiness, in KNN and Decision Tree models most correct and wrong predictions were made for happiness 
whilst most predictions for the Naïve Bayes model were for enthusiasm in which it made a lot of wrong predictions on tweets that reflect happiness instead it labelled 
them as enthusiastic. Moreover, I have noticed that KNN was the slowest model as it took about 0.15s to train and predict the testing data, while the other models took about 0.125s. 
This is not a massive difference, however, if the data was bigger it would be better to use other models than KNN if the focus is about the computation time.<br/>
__Error analysis and other observations:__<br/>
Based on the accuracy and computation times of the models I used, I concluded that the Decision Tree model had the best accuracy using the default parameters, 
however, other models are capable of increasing the accuracy which can be done by modifying the parameters invoked into the classification methods, providing more data for the models to learn from, 
or perhaps by doing more data preparation methods such as adding part of speech and lemmatization.
