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
<<<<<<< HEAD
=======

>>>>>>> 989c7e7cff7f21d9c8b01dd7b05706f146e32574

app = Flask(__name__)
CORS(app)  

load_dotenv()

<<<<<<< HEAD
# Check if API keys are loaded
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("Pinecone API key not found")
=======

>>>>>>> 989c7e7cff7f21d9c8b01dd7b05706f146e32574
pinecone.init(
    api_key=pinecone_api_key,
    environment=os.getenv('PINECONE_ENV')
)

<<<<<<< HEAD
# Text loading and splitting
=======

>>>>>>> 989c7e7cff7f21d9c8b01dd7b05706f146e32574
loader = TextLoader('data.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4) 
docs = text_splitter.split_documents(documents)

<<<<<<< HEAD
# Embedding and Pinecone index
=======

>>>>>>> 989c7e7cff7f21d9c8b01dd7b05706f146e32574
embeddings = HuggingFaceEmbeddings()
index_name = "arya-data-base"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

<<<<<<< HEAD
# HuggingFace LLM setup
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
endpoint_url = f"https://api-inference.huggingface.co/models/{repo_id}"

huggingface_api_token = os.getenv('HUGGING_FACE_API')
if not huggingface_api_token:
    raise ValueError("Hugging Face API token not found")

llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token=huggingface_api_token,
=======
docsearch = Pinecone.from_existing_index(index_name, embeddings)

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
endpoint_url = f"https://api-inference.huggingface.co/models/{repo_id}"


llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token=os.getenv('HUGGING_FACE_API'),
>>>>>>> 989c7e7cff7f21d9c8b01dd7b05706f146e32574
    temperature=0.8, 
    top_k=50 
)

<<<<<<< HEAD
# Prompt setup
=======
>>>>>>> 989c7e7cff7f21d9c8b01dd7b05706f146e32574
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

<<<<<<< HEAD
# Create RAG pipeline
=======

>>>>>>> 989c7e7cff7f21d9c8b01dd7b05706f146e32574
rag_chain = (
    {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
<<<<<<< HEAD

# Chatbot route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json.get('message', "")
        if not isinstance(user_input, str) or not user_input.strip():
            return jsonify({'error': 'Invalid input message provided'}), 400
=======
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'error': 'No input message provided'}), 400
>>>>>>> 989c7e7cff7f21d9c8b01dd7b05706f146e32574

        response = rag_chain.invoke(user_input)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
