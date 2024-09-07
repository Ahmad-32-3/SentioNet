import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import libraries
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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

# use padding to add digits and truncating to cutoff digits to meet uniformity
padded_sequences = pad_sequences(sequences, maxlen=maxLength, padding='post', truncating='post') 

# Ensure labels are numpy array
labels = np.array(labels)

# Split data into 20% testing and 80% training sets with consistent split using random_state=42.
x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Verify the shape of x_train and y_train before training
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

# set up model
model = Sequential()

# Embedding layer to convert into vectors of fixed size vectors
model.add(Embedding(input_dim=vocabSize, output_dim=128))

# LSTM layer with 64 units to process sequences
model.add(LSTM(units=64))

# Dense layer with 64 units and relu activation 
model.add(Dense(units=64, activation='relu'))

# Second dense layer with 64 units and relu activation 
model.add(Dense(units=32, activation='relu'))

# Dropout layer to prevent overfitting by randomly setting 20% of input units to 0 at each update
model.add(Dropout(rate=0.2))

# Output layer with 1 unit and sigmoid activation for binary classification (positive or negative sentiment)
model.add(Dense(units=1, activation='sigmoid'))

# Compile model with binary cross-entropy loss for 'loss' analysis and Adam as a sqishification function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=9, batch_size=32)

# evaluate 
model.evaluate(x_test, y_test, verbose=1)

# save model
model.save('tfmodel_sentiment.keras')

print("Model saved to disk")