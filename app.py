
import json
from flask import Flask, request, jsonify
from utils.pdf_processor import extract_text_chunks
from utils.index_builder import create_vector_index, query_vector_index
import config
import requests

app = Flask(__name__)

# Global variable for vector store
vector_store = None

# Load PDF and build vector index
def initialize_vector_index():
    global vector_store
    chunks = extract_text_chunks("data/document.pdf")
    vector_store = create_vector_index(chunks)

# Function to query Groq's LLaMA 3 (8B) model
def llama3_8b_query(query, context_text):
    payload = {
        "messages": [{
            "role": "user",
            "content": f"you are an assistant for question answering task. Use the following context to give the answer. \n\n Question: {query} \n\n context: {context_text}"
        }],
        "model": "llama3-8b-8192"  # Specify the LLaMA 3 model
    }
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(config.LLAMA3_8B_MODEL_ENDPOINT, headers=headers, json=payload)
    return response.json()

@app.route('/query', methods=['POST'])
def query_chatbot():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query not provided"}), 400
    
    # Perform vector search to get relevant context
    docs = query_vector_index(vector_store, user_query)
    context_text = " ".join([doc.page_content for doc in docs])
    
    # Forward query and context to Groq LLaMA 3 (8B)
    response = llama3_8b_query(user_query, context_text)
    res = response['choices'][0]['message']['content']

    # Parse the response from LLaMA 3 (8B)
    if res:
        answer = res
        return jsonify({"answer": answer})
    else:
        return jsonify({"error": "Failed to get response from LLaMA 3 (8B)"}), 500

if __name__ == '__main__':
    initialize_vector_index()
    app.run(debug=True)
