# scripts/chat_agent.py

import os
import faiss
import numpy as np
import json
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .transaction_analyzer import TransactionAnalyzer

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processed_data')
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'gst_faiss_index.bin')
CHUNKS_METADATA_PATH = os.path.join(OUTPUT_DIR, 'gst_chunks_metadata.json')
TRANSACTIONS_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'transactions.csv')


# --- Gemini Configuration ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GENERATION_MODEL_NAME = 'gemini-2.5-pro' # Or 'gemini-1.0-pro'
EMBEDDING_MODEL_NAME = 'models/embedding-001' # Ensure this matches what was used for ingestion

# Download NLTK resources (run once, if not already done by data_ingestion)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Utility Functions (Copied/Modified from data_ingestion.py for consistency) ---
def preprocess_text(text):
    """Cleans and preprocesses text for embedding."""
    text = text.lower()
    text = ' '.join(word for word in text.split() if word.isalnum() or word.isspace())
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    return " ".join(filtered_words)

def get_gemini_embedding(text):
    """Generates an embedding for the given text using Gemini's embedding model."""
    try:
        embedding = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type="RETRIEVAL_QUERY" # Important: Use RETRIEVAL_QUERY for user queries
        )['embedding']
        return np.array(embedding, dtype='float32')
    except Exception as e:
        print(f"Error generating embedding for query: {text[:50]}... Error: {e}")
        return None

# --- Retrieval System ---
class GSTRetriever:
    def __init__(self, faiss_index_path, chunks_metadata_path):
        print("Loading FAISS index and chunk metadata...")
        self.index = faiss.read_index(faiss_index_path)
        with open(chunks_metadata_path, 'r', encoding='utf-8') as f:
            self.chunks_metadata = json.load(f)
        print(f"Loaded {len(self.chunks_metadata)} chunks.")

    def retrieve_relevant_chunks(self, query_embedding, k=5):
        """
        Searches the FAISS index for the top-k most similar chunks.
        Returns the metadata and text of these chunks.
        """
        D, I = self.index.search(query_embedding.reshape(1, -1), k) # D: distances, I: indices
        
        relevant_chunks_info = []
        for i in I[0]: # I[0] contains the indices of the top-k nearest neighbors
            if i < len(self.chunks_metadata): # Ensure index is valid
                relevant_chunks_info.append(self.chunks_metadata[i])
        return relevant_chunks_info

# --- Generative Agent ---
class GSTAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        # REMOVED: self.chat = self.generation_model.start_chat(history=[])
        # REMOVED: self.transaction_analyzer = TransactionAnalyzer()
        print(f"Gemini {GENERATION_MODEL_NAME} model initialized for stateless operation.")

    def load_transaction_data(self, file_path):
        """
        Loads and processes transaction data from a file path.
        Returns a pandas DataFrame on success or an error string on failure.
        """
        try:
            # Create a temporary analyzer instance to load data
            analyzer = TransactionAnalyzer(file_path)
            if analyzer.transactions.empty:
                raise ValueError("No transactions were loaded from the file.")
            print(f"Successfully loaded {len(analyzer.transactions)} transactions for this request.")
            return analyzer.transactions # Return the DataFrame
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading transaction data: {e}")
            return str(e) # Return the error message

    def generate_response(self, user_query, transactions_df=None, num_retrieved_chunks=5):
        # 1. Preprocess user query
        processed_query = preprocess_text(user_query)

        # 2. Get embedding for the user query
        query_embedding = get_gemini_embedding(processed_query)
        if query_embedding is None:
            return "I'm sorry, I couldn't process your query for embedding. Please try again."

        # 3. Retrieve relevant chunks from GST forms
        relevant_chunks = self.retriever.retrieve_relevant_chunks(query_embedding, k=num_retrieved_chunks)
        
        context_parts = []
        if relevant_chunks:
            context_parts.append("--- GST Document Context ---")
            context_parts.extend([f"Source: {chunk['filename']} ({chunk['category']})\n{chunk['text']}" for chunk in relevant_chunks])
        else:
            context_parts.append("No highly relevant GST document context found for your query.")

        # 4. Include transaction analysis if a DataFrame is provided
        transaction_analysis_info = ""
        if transactions_df is not None and not transactions_df.empty:
            # Create a temporary analyzer instance with the provided data
            temp_analyzer = TransactionAnalyzer()
            temp_analyzer.transactions = transactions_df

            transaction_analysis_info = "\n--- User Transaction Data Analysis ---\n"
            transaction_analysis_info += temp_analyzer.get_summary() + "\n"
            
            if "itc" in user_query.lower() or "input tax credit" in user_query.lower() or "purchase" in user_query.lower():
                 transaction_analysis_info += temp_analyzer.find_potential_itc_opportunities() + "\n"
            
            transaction_list_string = temp_analyzer.transactions.to_string()
            transaction_analysis_info += "\n--- Full Transaction List ---\n"
            transaction_analysis_info += transaction_list_string

            # Add more conditional analysis here based on keywords in user_query
            # e.g., if "sales" or "GSTR-1" is in query, add sales specific analysis
            # e.g., if "composition" is in query, add threshold checks

        # 5. Construct the final prompt for Gemini
        
        system_instruction = (
            "You are an AI assistant specialized in Indian GST (Goods and Services Tax) for small business owners. "
            "You provide accurate, concise, and helpful information. "
            "You will be given a 'User Query', 'GST Document Context' (if available), and optionally 'User Transaction Data Analysis'. "
            "Use ALL provided context to answer the query. "
            "If transaction data is provided and relevant, incorporate its insights. "
            "When answering, cite the GST document source filename(s) if the information came from the 'GST Document Context'. "
            "For transaction data insights, mention that the information comes from 'your transaction data'."
            "If the answer is not available in the context, clearly state that you don't have enough information. "
            "Do not make up information. Always prioritize clarity and directness." 
            "- Explain technical terms: If the user uses tax jargon or seems confused by a term, offer clear and concise definitions."
            "- Try to present the data in the visually appealing format."
            "- Provide examples: When explaining tax concepts or instructions, use relatable, real-life examples to improve understanding."
            "- Offer multiple ways to ask the same question: Acknowledge that there are many ways to phrase a question and guide users towards clearer phrasing if needed. Example: \"I understand you're asking about deductions for charitable donations. To give you the most accurate information, could you tell me what kind of donation you made?\""
            "- Ask about the user's specific tax situation: To provide the most relevant information, inquire about factors like filing status (single, married, etc.), dependents, income type, and any unique circumstances that might apply."
            "- Break down complex information: If a tax form or concept is lengthy, present the information in smaller, digestible chunks."
            "- Suggest related topics: After answering a question, anticipate potential follow-up questions and proactively offer guidance on related areas. Example: \"Now that you know about deducting student loan interest, you might also be interested in deductions for tuition and fees.\""
            "- Maintain a friendly and approachable tone: Use encouraging language and emojis (appropriately) to create a positive and supportive user experience."
            "- Be mindful of user privacy: Never ask for sensitive personal information like aadhar or pan card numbers, bank account details, or exact income figures."
            "- Stay updated: Inform users that tax laws and regulations are subject to change, and encourage them to always refer to the most up-to-date information on the GST website."
            "- Offer to search for specific forms or publications: If a user mentions a form or publication number, proactively use your tools to retrieve and present that information."
            "- Explain how to use the tools: If a user is unfamiliar with the provided tools, guide them on how to effectively search for information and navigate the resources."
            "- If you don't know the answer, respond with \"I don't know.\""
        )

        full_context = "\n\n".join(context_parts) + transaction_analysis_info

        prompt = f"{system_instruction}\n\n" \
                 f"User Query: {user_query}\n\n" \
                 f"{full_context}\n\n" \
                 f"Based on the combined information provided above, answer the 'User Query'. " \
                 f"If the context does not contain enough information to fully answer, state that clearly. " \
                 f"Ensure your answer is actionable and relevant to a small business owner."

        # 6. Send the prompt to Gemini and get a response
        try:
            # Use generate_content for a stateless, single-turn conversation
            response = self.generation_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred while generating the response: {e}"

# --- Main Interaction Loop (This part is now for CLI testing only) ---
if __name__ == "__main__":
    # ... (This block can remain for local testing, but the API is the main entry point now)
    # ... It will now operate in a stateless manner per query.
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_METADATA_PATH):
        print("Error: FAISS index or chunk metadata files not found.")
        print("Please run scripts/data_ingestion.py first to generate them.")
        exit()

    retriever = GSTRetriever(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
    agent = GSTAgent(retriever)
    
    # For CLI mode, we can load data once and keep it in a local variable
    loaded_transactions_df = None
    default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'transactions.csv')
    if os.path.exists(default_file):
        print(f"AI: Attempting to load default transactions from '{os.path.basename(default_file)}'...")
        result = agent.load_transaction_data(default_file)
        if isinstance(result, pd.DataFrame):
            loaded_transactions_df = result
            print(f"AI: Transaction data loaded successfully.")
        else:
            print(f"AI: {result}")

    print("\nWelcome to the Indian GST AI Assistant! Ask me anything about GST.")
    print("Type 'exit' to quit the assistant.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the GST AI Assistant. Goodbye!")
            break
        else:
            # Pass the loaded dataframe with each call
            print(f"AI: {agent.generate_response(user_input, transactions_df=loaded_transactions_df)}")