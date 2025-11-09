import streamlit as st
import pickle
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Classifier")
st.markdown("""
This app predicts whether an email/message is **Spam** or **Ham** (not spam) 
using a trained Machine Learning model.
""")

@st.cache_resource
def load_models():
    try:
        with open('spam_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('standard_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, tfidf, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def preprocess_text(text):
    text = text.lower()
    
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        tokens = text.split()
    
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in stopwords.words('english') and w not in string.punctuation]
    ps = PorterStemmer()
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)

def predict_spam(text, model, tfidf, scaler):
    processed_text = preprocess_text(text)
    text_tfidf = tfidf.transform([processed_text]).toarray()
    text_scaled = scaler.transform(text_tfidf)
    prediction = model.predict(text_scaled)[0]
    probability = model.predict_proba(text_scaled)[0]
    return prediction, probability

def main():
    model, tfidf, scaler = load_models()
    
    if model is None:
        st.error("❌ Model files not found. Please make sure all .pkl files are uploaded.")
        st.info("""
        **Required files:**
        - spam_classifier_model.pkl
        - tfidf_vectorizer.pkl  
        - standard_scaler.pkl
        """)
        return
    
    st.subheader("Enter your email/message text:")
    user_input = st.text_area(
        "Paste the email or message content here:",
        height=200,
        placeholder="Type or paste your email content here..."
    )
    
    if st.button("Check for Spam", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    prediction, probability = predict_spam(user_input, model, tfidf, scaler)
                    
                    st.subheader("Results:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error("🚨 **SPAM**")
                            st.metric("Spam Probability", f"{probability[1]:.2%}")
                        else:
                            st.success("✅ **HAM**")
                            st.metric("Ham Probability", f"{probability[0]:.2%}")
                    
                    with col2:
                        spam_prob = probability[1]
                        st.progress(int(spam_prob * 100))
                        st.caption(f"Spam confidence: {spam_prob:.2%}")
                    
                    with st.expander("Detailed Probabilities"):
                        st.write(f"**Ham (Not Spam)**: {probability[0]:.4f} ({probability[0]:.2%})")
                        st.write(f"**Spam**: {probability[1]:.4f} ({probability[1]:.2%})")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please try again with different text.")
                
        else:
            st.warning("Please enter some text to analyze.")
    
    with st.expander("💡 Example messages to try"):
        st.write("**Spam examples:**")
        st.code("""
        Congratulations! You've won a $1000 Walmart gift card. 
        Click here to claim your prize now: http://bit.ly/winprize
        """)
        
        st.write("**Ham examples:**")
        st.code("""
        Hi John, just checking in about our meeting tomorrow at 2 PM. 
        Please let me know if you need to reschedule. Best, Sarah
        """)

if __name__ == "__main__":
    main()
