import joblib
import streamlit as st
import tempfile

# Function to load model and tokenizer from uploaded files
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_file, tokenizer_file):
    model = joblib.load(model_file)
    tokenizer = joblib.load(tokenizer_file)
    return model, tokenizer

# Main Streamlit app code
def main():
    st.title('Twitter Sentiment Analysis')

    model_file = st.file_uploader("Upload sentiment model (.pkl file)", type="pkl")
    tokenizer_file = st.file_uploader("Upload tokenizer (.pkl file)", type="pkl")

    if model_file and tokenizer_file:
        # Create temporary files to save uploaded files
        with tempfile.NamedTemporaryFile(delete=False) as tmp_model_file:
            tmp_model_file.write(model_file.read())
            model_path = tmp_model_file.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp_tokenizer_file:
            tmp_tokenizer_file.write(tokenizer_file.read())
            tokenizer_path = tmp_tokenizer_file.name

        model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

        text = st.text_input('Enter a tweet:')
        if st.button('Predict Sentiment'):
            # Example: tokenization and prediction
            tokens = tokenizer.tokenize(text)
            prediction = model.predict([tokens])  # Ensure the correct format
            st.write(f"Sentiment prediction: {prediction}")

if __name__ == '__main__':
    main()

