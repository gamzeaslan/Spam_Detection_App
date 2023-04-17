#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle 
import streamlit 


# In[2]:


import nltk 
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords

ps = PorterStemmer()
nltk.download('stopwords')


# In[3]:


def sms_transform(sms_text):  
    text = sms_text.lower()
    text = nltk.word_tokenize(text)    
    word = []   
    for i in text:
        if i.isalnum() and  i not in stopwords.words('english') and i not in string.punctuation:
            word.append(ps.stem(i))            
    return " ".join(word)  


# In[4]:


model = pickle.load(open('spam_detection_model.sav','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))


# In[5]:


streamlit.title("SMS Spam Deteciton")
input_sms = streamlit.text_area("Please enter sms text")

if streamlit.button('Spam or not spam'):

    # 1. preprocess
    sms_text = sms_transform(input_sms)
    # 2. vectorize
    vector = tfidf.transform([sms_text])
    # 3. predict
    result = model.predict(vector)[0]
    # 4. Display
    if result == 1:
        streamlit.header("Spam 😮")
    else:
        streamlit.header("Not Spam 😎")


# In[ ]:




