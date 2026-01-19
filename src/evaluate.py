import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import load_and_clean_data

MAX_LEN = 150
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# Load model & tokenizer
model = load_model("models/toxicity_model")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load test data
df_test = load_and_clean_data("data/test.csv")
X_test = df_test["comment_text"].values

# Tokenize & pad
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post")

# Predict
y_pred = model.predict(X_test_pad)

# Show predictions for first 5 samples
for i in range(5):
    print(f"\nComment {i+1}:")
    print(df_test["comment_text"].iloc[i][:200])
    for j, label in enumerate(labels):
        print(f"{label}: {y_pred[i][j]:.4f}")
