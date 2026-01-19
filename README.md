# Toxicity Detection using Deep Learning (NLP)

This project implements a **deep learning–based toxicity detection system** for identifying harmful online comments. It uses **Natural Language Processing (NLP)** techniques and compares the performance of **Bi-LSTM** and **CNN** models for multi-label toxic comment classification.
The system predicts the following categories:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

A **Streamlit web application** is included for real-time prediction and model comparison.

## Project Structure
toxicity-detection/
│
├── src/
│ ├── preprocess.py # Text cleaning & preprocessing
│ ├── train.py # Bi-LSTM model training
│ ├── train_cnn.py # CNN model training
│ ├── evaluate.py # LSTM evaluation on test data
│ ├── evaluate_cnn.py # CNN evaluation on test data
│ └── predict.py # Prediction utility
│
├── app.py # Streamlit web application
├── requirements.txt # Project dependencies
├── README.md
└── .gitignore

## Dataset

The project uses a toxic comments dataset with labeled categories:
- `train.csv` – used for training and validation  
- `test.csv` – used for model evaluation  
Text data is cleaned using regex, lowercasing, and noise removal before tokenization.

## Models Implemented

### 1️⃣ Bi-LSTM Model
- Embedding layer
- Bidirectional LSTM layers
- Dropout for regularization
- Sigmoid output for multi-label classification

### 2️⃣ CNN Model
- Embedding layer
- 1D Convolution layers
- Max pooling
- Dense output layer

Both models achieved **very similar performance** on the test dataset, showing strong capability for toxic content detection.

## Model Training

Train the models locally using:

```bash
python src/train.py        # Train Bi-LSTM
python src/train_cnn.py    # Train CNN
python src/evaluate.py        # Evaluate Bi-LSTM
python src/evaluate_cnn.py    # Evaluate CNN
streamlit run app.py
```
Features:
User text input
Real-time toxicity prediction
Option to switch between LSTM and CNN models

Technologies Used

Python
TensorFlow / Keras
Scikit-learn
NLTK
Streamlit
Pandas, NumPy
