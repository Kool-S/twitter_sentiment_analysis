import os
import joblib
import streamlit as st

# Paths to your locally stored model and tokenizer
model_path = 'C:/Users/Shristi/Downloads/sentiment_model.pkl'
tokenizer_path = 'C:/Users/Shristi/Downloads/tokenizer.pkl'

# Function to load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    # Load the model and tokenizer using joblib
    model = joblib.load(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer

# Main Streamlit app code
def main():
    st.title('Twitter Sentiment Analysis')
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Example sentiment analysis
    text = st.text_input('Enter a tweet:')
    if st.button('Predict Sentiment'):
        # Replace with your actual sentiment analysis logic
        # Example: tokenization and prediction
        tokens = tokenizer.tokenize(text)
        prediction = model.predict([tokens])  # Make sure to pass the tokens in the correct format
        st.write(f"Sentiment prediction: {prediction}")

if __name__ == '__main__':
    main()

