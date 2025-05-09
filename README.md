# SentioNet

## Overview

SentioNet is a text-based sentiment classification project built using **TensorFlow** and **Keras**.  
It leverages a **recurrent neural network (RNN)** with **LSTM architecture** to uncover long-term dependencies in text, boosting model accuracy and improving emotional context understanding.  
The project uses the IMDB dataset to classify reviews as positive or negative.

---

## Features

- **Recurrent Neural Network (RNN)** for sentiment analysis  
- **LSTM layers** to capture long-term dependencies in text  
- **L2 regularization** and **dropout** to reduce overfitting (~20% improvement)  
- Evaluated on **IMDB dataset** with ~15% accuracy boost  
- Preprocessing pipeline: tokenization, padding, train/test split

---

## Files

- `Neural_Network.py` → Builds, trains, and saves the LSTM model  
- `loadingmodel.tf` → Loads, evaluates, and re-saves the trained model

---

## Setup

1. Install required libraries:
    ```bash
    pip install pandas numpy scikit-learn tensorflow
    ```

2. Place the IMDB dataset CSV at:
    ```
    D:\Python\IMDB Dataset.csv
    ```

3. Run the training script:
    ```bash
    python Neural_Network.py
    ```

4. Run the evaluation script:
    ```bash
    python loadingmodel.tf
    ```

---

## Model Architecture

- **Embedding layer** → 5000-word vocab, 128-dim vectors  
- **LSTM (64 units)** → Sequence processing  
- **Dense layers (64, 32 units, relu)** → Feature extraction  
- **Dropout (0.2)** → Regularization  
- **Output layer (sigmoid)** → Binary classification

---


