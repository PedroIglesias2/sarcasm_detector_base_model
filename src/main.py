#!/usr/bin/env python
# coding: utf-8

# # Sarcasm Detector

# ## Get and Load Data

# In[2]:


import contractions
from bs4 import BeautifulSoup
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


# In[3]:


df = pd.read_json('../data/processed/SarcasmDetect.json', lines=True)
df.head()


# ## Remove all records with no headline text. Clean and split data

# In[4]:


df = df[df['headline'] != '']
df.info()


# In[5]:


random_state=42 # for reproducibility


# In[6]:


df['is_sarcastic'].value_counts() # see data distribution


# In[7]:


# Split the data.
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['is_sarcastic', 'article_link']), df['is_sarcastic'], test_size=0.3, random_state=42, stratify =df['is_sarcastic'])
X_train.shape, X_test.shape


# In[8]:


from collections import Counter
Counter(y_train), Counter(y_test) #check split is correct


# In[9]:


X_train.head()


# In[10]:


X_train["headline"].iloc[0]


# In[11]:


y_train


# In[16]:


# cleaning text auxilary functions
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in docs:
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc, flags=re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        norm_docs.append(doc)
    return norm_docs


# In[17]:


# Clean the data

norm_train_texts = pre_process_corpus(X_train['headline'].values)
norm_test_texts = pre_process_corpus(X_test['headline'].values)


# ## We build base line logistic regression model

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=False, min_df=2, max_df=1.0)

cv_train_features = cv.fit_transform(norm_train_texts)
cv_test_features = cv.transform(norm_test_texts)
print('BOW model:> Train features shape:', cv_train_features.shape, ' Test features shape:', cv_test_features.shape)


# In[19]:


# Logistic Regression model on BOW features
from sklearn.linear_model import LogisticRegression

# instantiate model
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, solver='lbfgs', random_state=42)

# train model
lr.fit(cv_train_features, y_train)

# predict on test data
lr_bow_predictions = lr.predict(cv_test_features)


# In[20]:


# Test model on test data
print(classification_report(y_test, lr_bow_predictions))
pd.DataFrame(confusion_matrix(y_test, lr_bow_predictions))

