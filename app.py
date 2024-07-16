
import streamlit as st
import os
import numpy as np
import torch
print("Transformers importing!")
from transformers import AutoTokenizer, AutoModel
print("Transformers imported successfully!")
from PyPDF2 import PdfReader
import docx
import google.generativeai as genai

genai.configure(api_key='AIzaSyDvWSxEts3K3bE_pOFvhXX2vTnZxOetQO8')
# Function to extract text from various file types
# Function to extract text from various file types
def extract_text(file_path):
    _, ext = os.path.splitext(file_path)
    text = ""
    
    if ext == ".txt":
        with open(file_path, 'r') as f:
            text = f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    elif ext == ".docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError("Unsupported file type: {}".format(ext))
    
    return text

# Load tokenizer and model for embeddings only once
@st.cache_resource
def load_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
@st.cache_resource
def get_embeddings(sentences):
    embeddings = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence.strip(), return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1))
    return embeddings

# Streamlit app
st.title("Text Embedding and GenAI Query")

uploaded_file = st.file_uploader("Upload a file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
query = st.text_input("Enter your query")

# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

if st.button("Submit"):
    if uploaded_file is not None and query:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        text = extract_text(file_path)
        sentences = text.split('.')
        sentence_embeddings = get_embeddings(sentences)
        sentence_embeddings = [emb.squeeze().numpy() for emb in sentence_embeddings]
        
        # Prepare text prompt for the GenAI model
        text_prompt = f"Query: {query}\nRelevant Text: {text}"
        
        # Configure and call the GenAI model
        if 'genai_model' not in st.session_state:
            
            st.session_state.genai_model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = st.session_state.genai_model.generate_content(text_prompt)
        result = response.text
        st.write("Response from GenAI:")
        st.write(result)
    else:
        st.warning("Please upload a file and enter a query.")