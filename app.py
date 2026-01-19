import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocess import clean_text

MAX_LEN = 150
labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

st.set_page_config(page_title="Toxicity Detection", layout="centered")
st.title("Toxic Comment Detection")
st.write("Compare **LSTM vs CNN** models")

# Model selector
model_choice = st.selectbox(
    "Select Model",
    ["LSTM", "CNN"]
)

@st.cache_resource
def load_selected_model(choice):
    if choice == "LSTM":
        model = load_model("models/toxicity_model")
        with open("models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
    else:
        model = load_model("models/toxicity_cnn")
        with open("models/tokenizer_cnn.pkl", "rb") as f:
            tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_selected_model(model_choice)

text = st.text_area("Enter a comment")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        clean = clean_text(text)
        seq = tokenizer.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        preds = model.predict(pad)[0]

        st.subheader(f"Results using {model_choice}")
        for label, score in zip(labels, preds):
            st.write(f"**{label}** : {score:.4f}")
