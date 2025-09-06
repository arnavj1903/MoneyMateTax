from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from scripts.chat_agent import GSTAgent, GSTRetriever
from waitress import serve

# --- Configuration ---
OUTPUT_DIR = 'processed_data'
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'gst_faiss_index.bin')
CHUNKS_METADATA_PATH = os.path.join(OUTPUT_DIR, 'gst_chunks_metadata.json')
# Define the path to your single transaction file, assuming it's in the root
TRANSACTIONS_FILE_PATH = 'transactions.csv' 

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Initialize the GST Agent (Global) ---
print("Initializing GST Agent...")
if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_METADATA_PATH):
    raise FileNotFoundError("FAISS index or metadata not found. Run data_ingestion.py first.")
retriever = GSTRetriever(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
agent = GSTAgent(retriever)
print("GST Agent initialized successfully.")

# --- Load Transaction Data Globally at Startup ---
GLOBAL_TRANSACTIONS_DF = None
print(f"Attempting to load global transaction file from: {TRANSACTIONS_FILE_PATH}")
if os.path.exists(TRANSACTIONS_FILE_PATH):
    result = agent.load_transaction_data(TRANSACTIONS_FILE_PATH)
    if isinstance(result, pd.DataFrame):
        GLOBAL_TRANSACTIONS_DF = result
        print(f"✅ Successfully loaded {len(GLOBAL_TRANSACTIONS_DF)} transactions globally.")
    else:
        print(f"❌ ERROR loading global transactions: {result}")
else:
    print(f"⚠️ WARNING: Global transaction file not found at '{TRANSACTIONS_FILE_PATH}'. The agent will not have transaction context.")


@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    user_input = data['message']
    
    print(f"[INFO] Received query: '{user_input}'. Using global transaction data.")

    # Generate response, always passing the globally loaded dataframe
    response_text = agent.generate_response(user_input, transactions_df=GLOBAL_TRANSACTIONS_DF)

    return jsonify({'reply': response_text})

# --- Run the Server ---
if __name__ == '__main__':
    print("Starting production server with Waitress on http://0.0.0.0:5000")
    serve(app, host='0.0.0.0', port=5000)