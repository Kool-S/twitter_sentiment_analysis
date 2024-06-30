import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set page title
st.set_page_config(page_title="Sentiment Analysis App")

@st.cache_resource
def load_model():
    try:
        # Try loading from pickle
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        model = model.to('cpu')
    except:
        # If pickle fails, load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return model

@st.cache_resource
def load_tokenizer():
    try:
        # Try loading from pickle
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    except:
        # If pickle fails, load pre-trained tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1)
    return "Positive" if prediction.item() == 1 else "Negative"

# Streamlit app
st.title('Sentiment Analysis App')

user_input = st.text_area("Enter your text here:")

if st.button('Predict Sentiment'):
    if user_input:
        with st.spinner('Analyzing sentiment...'):
            result = predict_sentiment(user_input)
        st.write(f"The sentiment of the text is: {result}")
    else:
        st.warning("Please enter some text for analysis.")

# Add some information about the app
st.markdown("""
### About this app
This app uses a pre-trained model to predict the sentiment of input text.
The model classifies text as either positive or negative.

### How to use
1. Enter your text in the text area above.
2. Click the 'Predict Sentiment' button.
3. The app will display whether the sentiment is positive or negative.
""")


