import streamlit as st
from transformers import pipeline
import pytesseract
from PIL import Image
import os
from sentence_transformers import SentenceTransformer, util

# Load Hugging Face token from secrets
hf_token = st.secrets["hf_token"]

# Load OCR model, NLP pipeline, and semantic search model
ocr_model = pytesseract
nlp_pipeline = pipeline("text-classification", model="google-bert/bert-base-uncased", tokenizer="google-bert/bert-base-uncased", use_auth_token=hf_token)

# Semantic search model
semantic_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

# Application title
st.title("OCR, NLP, and Semantic Search App")

# Upload image for OCR
st.subheader("Upload an Image for Text Extraction (OCR)")
uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Use OCR to extract text from the image
    image = Image.open(uploaded_image)
    extracted_text = ocr_model.image_to_string(image)
    
    # Display the extracted text
    st.subheader("Extracted Text from Image:")
    st.write(extracted_text)

    # Process the extracted text using NLP model
    if extracted_text.strip():
        prediction = nlp_pipeline(extracted_text)
        st.subheader("Text Classification Result:")
        st.write(prediction)

# Input text for semantic matching
st.subheader("Enter Text for Semantic Matching")
input_text = st.text_area("Enter text to search for semantic similarity:")

if input_text:
    # Calculate similarity between extracted text and input text
    embeddings = semantic_model.encode([extracted_text, input_text])
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    
    # Display similarity score
    st.subheader("Semantic Similarity Score:")
    st.write(f"Cosine Similarity: {similarity_score.item():.4f}")
