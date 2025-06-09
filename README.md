# Disaster Tweets Classification using Natural Language Processing (NLP)

This project involves building a machine learning pipeline to classify tweets as real disaster-related or not using advanced NLP techniques. The work was based on a Kaggle competition dataset and focuses on improving disaster response using social media analytics.

## Problem Statement

During natural disasters or crises, people post real-time updates on social media. Distinguishing real disaster tweets from irrelevant ones helps emergency responders, governments, and news agencies react more effectively. This project applies NLP techniques to predict whether a tweet refers to a real disaster event.

## Dataset

- Source: [Kaggle - Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- Size: ~10,000 tweets
- Label: `target` = 1 (disaster), 0 (non-disaster)
- Includes metadata such as tweet text, location, and keyword

## Exploratory Data Analysis

Key insights:
- Tweets related to disasters tend to be longer (>125 characters)
- Class imbalance is moderate; no major oversampling needed
- Most disaster tweets originate from the USA, followed by UK, Africa, and India
- High-frequency words for disaster tweets include “fire”, “kill”; for non-disasters: “like”, “get”

## NLP Preprocessing Steps

1. Text Segmentation
2. Tokenization
3. Stop Words Removal
4. Stemming and Lemmatization
5. POS Tagging
6. Named Entity Recognition (NER)

## Word Embeddings

We experimented with the following:
- **GloVe**: Static word vectors; lacks contextual understanding
- **BERT**: Contextualized embeddings for deeper language comprehension

## Modeling Techniques

Used deep learning models to handle sequential and contextual text data:
- LSTM (Long Short-Term Memory networks)
- Embedding combinations:
  - GloVe + LSTM
  - BERT + LSTM
  - Stemming + Lemmatization + BERT + LSTM

## Model Evaluation

Performance Metrics:
- Accuracy
- Precision, Recall, F1-Score
- Kaggle public leaderboard score for final submission

Best performing setup:
- BERT + LSTM with lemmatized text
- Achieved high classification accuracy and good generalization on unseen data

## Summary and Business Impact

- Enables fast, reliable classification of real disaster tweets
- Enhances response times during emergencies
- Helps filter noise from social media streams for government and aid agencies
- Scalable and deployable in real-time monitoring tools

## Tech Stack

- Python (NLTK, pandas, NumPy)
- TensorFlow/Keras
- Hugging Face Transformers (BERT)
- GloVe Embeddings
- Jupyter Notebooks

## Repository Structure

