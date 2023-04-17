# Project Overview
* With this project, you can get information on whether sms messages are spam.
* The dataset used for the model includes sms messages and whether these messages are spam or not.
* Built a client facing API using streamlit


# Code and Resources Used:
* **Python Version** : 3.10.9
* **Packages** : pandas ,matplotlib,sklearn,pickle,streamlit,warnings ,nltk ,plotly and wordcloud

# Data Cleaning
* With NLP, sms messages were separated into words, those containing only alphanumeric characters, those without punctuation marks, and those without English stopwords were taken and these words were added to the dataset as a new column with word roots using PorterStemmer.

# EDA
* At this stage, an interactive pie chart was drawn to see the percentages of spam and non-spam messages in the data set.
![alt text](https://github.com/gamzeaslan/spam_detection/blob/main/pie.png "Pie Graph")
* In conclusion, we can say that the dataset contains bias as there is a large difference between the percentages of spam and non-spam messages.

* I used Word Cloud to visualize the most frequent words in spam messages and non-spam messages
![alt text](https://github.com/gamzeaslan/spam_detection/blob/main/spam_wc.png "Spam WordCloud")
![alt text](https://github.com/gamzeaslan/spam_detection/blob/main/non_spam_wc.png "Non-Spam WordCloud")

# Model Building
* The frequencies of the words were calculated using the CountVectorizer library. The weights of the words were calculated with the TfidfVectorizer
* Then, as a result of the fit and estimation processes using BernoulliNB, the accuracy value of the model was obtained as 0.96.
![alt text](https://github.com/gamzeaslan/spam_detection/blob/main/classification_report.png "Classification Report")
![alt text](https://github.com/gamzeaslan/spam_detection/blob/main/confusion_matrix.png "Confusion Matrix")

# APP
* By using Streamlit, it is estimated whether the sms texts entered by the user are spam or not through the pre-recorded model
![alt text](https://github.com/gamzeaslan/spam_detection/blob/main/app.png "App")
