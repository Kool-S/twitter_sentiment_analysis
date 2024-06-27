import os
import gdown
import joblib
import streamlit as st

# URLs to your Google Drive hosted files
model_url = 'https://drive.google.com/file/d/1oxwbmuuuMpwNOP8E8y9oppGJHqMtNY8q/view?usp=drive_link'
tokenizer_url = 'https://drive.google.com/file/d/15EVgcq5NLzYzkWLVLhFaVf_n2ZVtNsd8/view?usp=drive_link'

# Function to download files if not already downloaded
def download_files():
    if not os.path.exists('sentiment_model.pkl'):
        gdown.download(model_url, 'sentiment_model.pkl', quiet=False)
    if not os.path.exists('tokenizer.pkl'):
        gdown.download(tokenizer_url, 'tokenizer.pkl', quiet=False)

# Load model and tokenizer function
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    # Download files if not already downloaded
    download_files()
    
    # Load the model and tokenizer using joblib (or appropriate method)
    model = joblib.load('sentiment_model.pkl')
    tokenizer = joblib.load('tokenizer.pkl')
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
        prediction = model.predict(tokens)
        st.write(f"Sentiment prediction: {prediction}")

if __name__ == '__main__':
    main()

