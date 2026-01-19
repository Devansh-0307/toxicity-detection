import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import load_and_clean_data

MAX_LEN = 150

# Load CNN model
model = load_model("models/toxicity_cnn")

# Load tokenizer
with open("models/tokenizer_cnn.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load test dataset (NO labels here)
df_test = load_and_clean_data("data/test.csv")

X_test = df_test["comment_text"].values

# Tokenize & pad
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post")

# Predict
preds = model.predict(X_test_pad)

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# Print first 5 predictions (same as LSTM)
for i in range(5):
    print(f"\nComment {i+1}:")
    print(X_test[i][:200])  # preview
    for j, label in enumerate(labels):
        print(f"{label}: {preds[i][j]:.4f}")
