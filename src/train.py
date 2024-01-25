#!/usr/bin/env python
# coding: utf-8

# # Sarcasm Detector

# ## Get and Load Data

from dill import dump, load
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os
import dotenv
import nltk
from src.data.make_dataset import main as make_dataset
from src.data.make_dataset import pre_process_corpus
from src.data.make_dataset import read_train_data

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


# Load data
project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)
data_path = os.getenv("DATA_PATH")
processed_path = os.getenv("PROCESSED_PATH")


# # ## Remove all records with no headline text. Clean and split data
# df = df[df['headline'] != '']

random_state=42 # for reproducibility

# Clean the data
X_train, y_train = read_train_data(processed_path)
norm_train_texts = pre_process_corpus(X_train['headline'].values)


# ## We build base line logistic regression model
cv = CountVectorizer(binary=False, min_df=2, max_df=1.0)
cv_train_features = cv.fit_transform(norm_train_texts)
# Logistic Regression model on BOW features
# instantiate model
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, solver='lbfgs', random_state=42)
# train model
lr.fit(cv_train_features, y_train)

model_path = os.getenv("MODEL_PATH")
if not os.path.exists(model_path):
    os.mkdir(model_path)
model_file_name = model_path + "/LogRegression.pkl"

with open(model_file_name, "wb") as f:
    dump((lr,cv), f)

    
    

