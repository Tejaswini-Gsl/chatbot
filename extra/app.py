import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="PDF QA System", layout="wide")

# Initialize session state for storing the QA chain
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

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
    
    # Split text
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
    
    # Initialize LLM
    llm = LlamaCpp(
        model_path="mistral-7b-v0.1.Q4_K_M.gguf",
        n_ctx=2048,
        temperature=0.3,
        max_tokens=1000,
        verbose=True
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_separator": "\n\n",
        },
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain

# Streamlit UI
st.title("📚 PDF Question Answering System")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'])

if uploaded_file is not None:
    if st.session_state.qa_chain is None:
        with st.spinner('Initializing the QA system...'):
            st.session_state.qa_chain = initialize_qa_system(uploaded_file)
        st.success('QA system is ready!')

    # Question input
    question = st.text_input("Ask a question about your PDF:")
    
    if question:
        with st.spinner('Finding answer...'):
            result = st.session_state.qa_chain({"query": question})
            
            # Post-process answer to ensure complete sentences
            answer = result['result']
            if not answer.endswith(('.', '!', '?', ':', ')', ']', '}')):
                last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
                if last_period != -1:
                    answer = answer[:last_period + 1]
            
            # Display answer in a nice format
            st.markdown("### Answer:")
            st.write(answer)
            
            # Option to view source documents
            # with st.expander("View Source Documents"):
            #     for i, doc in enumerate(result['source_documents']):
            #         st.markdown(f"**Source {i+1}:**")
            #         st.write(doc.page_content)
            #         st.markdown("---")
else:
    st.info("Please upload a PDF document to get started.")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using LangChain and Streamlit")