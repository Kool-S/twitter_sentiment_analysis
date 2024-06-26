import numpy as np
import pandas as pd
import os
import streamlit as st
import joblib

# List input data files (optional for debugging purposes)
for dirname, _, filenames in os.walk('/app/twitter_sentiment_analysis/models'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load model and tokenizer
model_path = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'models', 'tokenizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Function to predict sentiment
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Streamlit app
st.title("Twitter Sentiment Analysis")
st.write("Enter the text you want to analyse:")

user_input = st.text_area("")

if st.button("Analyze"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text for analysis")

