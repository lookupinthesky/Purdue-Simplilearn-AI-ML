# Twitter Hate Speech Detection

## Problem Statement

Twitter is the biggest platform where anybody and everybody can have their views heard. Some of these voices spread hate and negativity. Twitter is wary of its platform being used as a medium  to spread hate. 

You are a data scientist at Twitter, and you will help Twitter in identifying the tweets with hate speech and removing them from the platform. You will use NLP techniques, perform specific cleanup for tweets data, and make a robust model.

Domain: Social Media

Analysis to be done: Clean up tweets and build a classification model by using NLP techniques, cleanup specific for tweets data, regularization and hyperparameter tuning using stratified k-fold and cross validation to get the best model.

Content: 

id: identifier number of the tweet

Label: 0 (non-hate) /1 (hate)

Tweet: the text in the tweet


## Solution

### 1. Preprocessing and EDA

View the code on [GitHub](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Twitter%20Sentiment%20Analysis/twitter-hate-speech-preprocessing-and-eda.ipynb)

### 2. Model Building using CNN - LSTM architectures

[Model Building - 1 - Jupyter Notebook](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Twitter%20Sentiment%20Analysis/twitter-hate-speech-model-building.ipynb)

[Model Building - 2 - Jupyter Notebook](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Twitter%20Sentiment%20Analysis/twitter-hate-speech-model-building%202.ipynb)

### Model Deployment

Flask app packaged in Docker Container deployed on Heroku

[See live Demo of Proof of Concept](https://www.twitterdocksentiment.herokuapp.com)

[See code on Github](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/tree/deployment-heroku/Twitter%20Sentiment%20Analysis)
 

