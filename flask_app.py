from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import pinecone
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
app = Flask(__name__)
CORS(app)  

load_dotenv()

# Check if API keys are loaded
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
import pinecone
import os

# Initialize Pinecone
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = "us-west1-gcp"  # Set this to your correct environment

# Ensure API key is provided
if not pinecone_api_key:
    raise ValueError("Pinecone API key not found.")

# Create the Pinecone client


# Create the index client
index_name = "arya-data-base"
pc = pinecone.Index(index_name)

# Check if the index exists in the list of indexes
if index_name not in pinecone.list_indexes():
    raise ValueError(f"Index '{index_name}' does not exist.")


loader = TextLoader('data.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4) 
docs = text_splitter.split_documents(documents)

# Embedding and Pinecone index
embeddings = HuggingFaceEmbeddings()
index_name = "arya-data-base"
index = pc.Index(index_name)
docsearch = PineconeVectorStore(index=index, embedding=embeddings)

# HuggingFace LLM setup
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
endpoint_url = f"https://api-inference.huggingface.co/models/{repo_id}"

huggingface_api_token = os.getenv('HUGGING_FACE_API')

if not huggingface_api_token:
    raise ValueError("Hugging Face API token not found")

llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token=huggingface_api_token,
    temperature=0.8, 
    top_k=50 
)

template = """
You are Arya, the official bot of Arya Bhatt Hostel. Humans will ask you questions about the Arya Bhatt Hostel. 
Use the following context to answer the question from the vector database only. 
If you don't know the answer, just say you don't know. 
Keep the answer within 1 sentence and concise.

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    template=template, 
    input_variables=["question"]
)

# Create RAG pipeline
rag_chain = (
    {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chatbot route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json.get('message', "")
        if not isinstance(user_input, str) or not user_input.strip():
            return jsonify({'error': 'Invalid input message provided'}), 400

        # Instead of invoking RAG chain, return a mock response
        return jsonify({'response': f'Mock response to: {user_input}'})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
