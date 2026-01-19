import pickle
import  numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

MAX_LEN = 150
model = load_model("models/toxicity_model")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
def predict_toxicity(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq,maxlen=MAX_LEN,padding="post")
    preds = model.predict(pad)[0]
    return dict(zip(labels,preds))

if __name__ == "__main__":
    sample = "You are a stupid idiot"
    result = predict_toxicity(sample)
    for k,v in result.items():
        print(f"{k}:{v:.4f}")