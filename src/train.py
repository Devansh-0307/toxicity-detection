import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import load_and_clean_data


MAX_WORDS = 50000
MAX_LEN = 150
EMBEDDING_DIM = 128

df = load_and_clean_data('data/train.csv')
x = df["comment_text"].values
y = df[
    ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
].values
x_train,x_val,y_train,y_val = train_test_split(
    x,y,test_size=0.2,random_state=42
)
tokenizer = Tokenizer(num_words=MAX_WORDS,oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_val_seq = tokenizer.texts_to_sequences(x_val)

x_train_pad = pad_sequences(x_train_seq,maxlen=MAX_LEN,padding="post")
x_val_pad = pad_sequences(x_val_seq,maxlen=MAX_LEN,padding="post")

model = Sequential([
    Embedding(MAX_WORDS,EMBEDDING_DIM,input_length = MAX_LEN),
    Bidirectional(LSTM(64,return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(6,activation='sigmoid')
])

model.compile(
    loss="binary_crossentropy",
    optimizer = "adam",
    metrics = ["accuracy"]
)

model.summary()
early_stop = EarlyStopping(
    monitor = "val_loss",
    patience = 2,
    restore_best_weights = True
)

model.fit(
    x_train_pad,
    y_train,
    validation_data=(x_val_pad,y_val),
    epochs = 5,
    batch_size = 128,
    callbacks = [early_stop]
)

model.save("models/toxicity_model", save_format="tf")

import pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
