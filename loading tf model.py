import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import libraries
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Load the model
model = load_model('tfmodel_sentiment.keras')

# Print model summary to verify it has loaded
model.summary()

# get Data
sentimentData = pd.read_csv("D:\\Python\\IMDB Dataset.csv")

# Preprocessing the data
reviews = sentimentData['review'].values # Extract reviews 
labels = sentimentData['sentiment'].values # Extract 'positive' and 'negative' labels

labels = [1 if label == 'positive' else 0 for label in labels] # sets positive to 1 and negative to 0

vocabSize = 5000  # Define vocabulary size

tokenizer = Tokenizer(num_words=5000)  # Initialize a tokenizer, limit to 5000 words
tokenizer.fit_on_texts(reviews)        # Fit tokenizer on text data
sequences = tokenizer.texts_to_sequences(reviews)  # Convert text to sequences of integers

maxLength = 200  # Define the maximum length of sequences

# use padding to add digits and truncating to cutoff digits to meet uniformit
padded_sequences = pad_sequences(sequences, maxlen=maxLength, padding='post', truncating='post') 

# Ensure labels are numpy array
labels = np.array(labels)

# Split data into 20% testing and 80% training sets with consistent split using random_state=42.
x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# evaluate 
model.evaluate(x_test, y_test, verbose=1)

# save model
model.save('tfmodel_sentiment.keras')

print("Model saved to disk")
