# scripts/data_ingestion.py

import os
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Download NLTK resources (run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

load_dotenv() # Load environment variables from .env file

# --- Configuration ---
FORMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'forms')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'gst_faiss_index.bin')
CHUNKS_METADATA_PATH = os.path.join(OUTPUT_DIR, 'gst_chunks_metadata.json')

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Text Extraction ---
def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def load_all_gst_forms(root_dir):
    """Loads text from all PDF forms in the given root directory."""
    documents = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(folder_path, filename)
                    print(f"Processing {pdf_path}...")
                    content = extract_text_from_pdf(pdf_path)
                    if content:
                        documents.append({
                            'filename': filename,
                            'category': folder_name,
                            'content': content
                        })
    return documents

# --- Text Preprocessing ---
def preprocess_text(text):
    """Cleans and preprocesses text."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers (you might want to keep some numbers for forms)
    # For now, let's keep numbers. Remove only non-alphanumeric except spaces.
    text = ' '.join(word for word in text.split() if word.isalnum() or word.isspace())

    # Tokenize words
    words = word_tokenize(text)

    # Remove stopwords and single character words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]

    return " ".join(filtered_words)

# --- Text Chunking ---
def chunk_text(text, max_chunk_size=500, overlap=50):
    """Splits text into chunks with optional overlap."""
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            # Start the next chunk with an overlap
            current_chunk = current_chunk[max_chunk_size - overlap:]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --- Main Ingestion Process ---
def ingest_data():
    print("Starting data ingestion and preparation...")
    raw_documents = load_all_gst_forms(FORMS_DIR)
    processed_chunks = []
    
    for doc in raw_documents:
        preprocessed_content = preprocess_text(doc['content'])
        chunks = chunk_text(preprocessed_content)
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'id': f"{doc['filename']}_{doc['category']}_{i}",
                'filename': doc['filename'],
                'category': doc['category'],
                'chunk_index': i,
                'text': chunk
            })
    
    print(f"Extracted and chunked {len(processed_chunks)} total text chunks.")
    return processed_chunks

# Function to get embeddings from Gemini
def get_gemini_embedding(text):
    """Generates an embedding for the given text using Gemini's embedding model."""
    try:
        # Explicitly pass the full model name string to embed_content
        embedding = genai.embed_content(
            model="models/embedding-001", # <--- Make sure this is "models/embedding-001"
            content=text,
            task_type="RETRIEVAL_DOCUMENT" # Specify task type for better embeddings in RAG
        )['embedding']
        return np.array(embedding, dtype='float32')
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}... Error: {e}")
        return None

if __name__ == "__main__":
    processed_chunks = ingest_data()
    
    print("\nGenerating embeddings and building FAISS index...")
    
    embeddings = []
    chunk_metadata = []

    for i, chunk_data in enumerate(processed_chunks):
        print(f"Generating embedding for chunk {i+1}/{len(processed_chunks)}...")
        embedding = get_gemini_embedding(chunk_data['text'])
        if embedding is not None:
            embeddings.append(embedding)
            chunk_metadata.append({
                'id': chunk_data['id'],
                'filename': chunk_data['filename'],
                'category': chunk_data['category'],
                'chunk_index': chunk_data['chunk_index'],
                'text': chunk_data['text']
            })
        else:
            print(f"Skipping chunk {chunk_data['id']} due to embedding error.")

    if not embeddings:
        print("No embeddings generated. Exiting.")
        exit()

    embeddings_array = np.array(embeddings)
    embedding_dimension = embeddings_array.shape[1]

    # Build FAISS index
    index = faiss.IndexFlatL2(embedding_dimension) # L2 distance is common for similarity
    index.add(embeddings_array)

    # Save the FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    # Save chunk metadata to a JSON file
    with open(CHUNKS_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=4)
    print(f"Chunk metadata saved to {CHUNKS_METADATA_PATH}")

    print("\nData ingestion and preparation complete!")
    print(f"Total chunks indexed: {len(chunk_metadata)}")