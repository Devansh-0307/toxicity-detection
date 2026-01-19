import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+","",text)
    text = re.sub(r"<.*?>","",text)
    text = re.sub(r"[^a-z\s]","",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text
def load_and_clean_data(path):
    df = pd.read_csv(path)
    df['comment_text'] = df['comment_text'].apply(clean_text)
    return df