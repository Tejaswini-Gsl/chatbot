import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# import chromadb.api

# chromadb.api.client.SharedSystemClient.clear_system_cache()

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="📚 PDF Q&A Chat", layout="wide")

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Stores Q&A pairs

def initialize_qa_system(pdf_file):
    # Create persist directory
    PERSIST_PATH = "./persistentdb/"
    os.makedirs(PERSIST_PATH, exist_ok=True)
    
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    
    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    )
    split_docs = text_splitter.split_documents(documents)
    
    # Initialize embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_function,
        persist_directory=PERSIST_PATH
    )
    
    # Configure retriever
    retriever = db.as_retriever(
        search_kwargs={"k": 2}
    )
    
    # Create prompt template
    prompt_template = """
    You are a helpful AI assistant. You're tasked to Answer the question based on the context below. Be concise and complete without any repetition of sentences.
    Context: {context}
    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Initialize Gemini
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PROMPT,
        },
        return_source_documents=True,
    )
    
    return qa_chain, db

# Function to clean up resources
def cleanup_resources():
    if st.session_state.db is not None:
        st.session_state.db = None
    if st.session_state.qa_chain is not None:
        st.session_state.qa_chain = None
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")
    if os.path.exists("./persistentdb"):
        import shutil
        shutil.rmtree("./persistentdb")

# Streamlit UI
st.title("PDF Chat-based Q&A System")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'], on_change=cleanup_resources)

if uploaded_file is not None:
    if st.session_state.qa_chain is None:
        with st.spinner('Initializing the QA system...'):
            st.session_state.qa_chain, st.session_state.db = initialize_qa_system(uploaded_file)
        st.success('QA system is ready!')

    # Display Chat Messages
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**{chat['question']}**")
        with st.chat_message("assistant"):
            st.markdown(f"{chat['answer']}")

    # Input for new questions
    question = st.chat_input("Ask a question about your PDF:")

    if question:
        try:
            with st.spinner('Finding answer...'):
                result = st.session_state.qa_chain({"query": question})

                # Get and format the answer
                answer = result['result']

                # Append Q&A to chat history
                st.session_state.chat_history.append({"question": question, "answer": answer})

                # Display new question and answer dynamically
                with st.chat_message("user"):
                    st.markdown(f"**{question}**")
                with st.chat_message("assistant"):
                    st.markdown(f"{answer}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.qa_chain, st.session_state.db = initialize_qa_system(uploaded_file)

    # Button to clear chat history
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

else:
    st.info("Please upload a PDF document to get started.")

# Add footer
# st.markdown("---")
# st.markdown("🚀 Made with ❤️ using LangChain, Gemini, and Streamlit")
