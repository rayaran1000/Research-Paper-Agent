import os
import numpy as np
import faiss
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from Data_Ingestion import DataIngestion

# Environment variables
from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

## Load the Langchain API key
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


# Initialize FAISS index
dimension = 512  
index = faiss.IndexFlatL2(dimension)
#vector_store = FAISS(index,HuggingFaceBgeEmbeddings())


# Streamlit app
st.title("Research Paper Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

data_ingestion = DataIngestion()

if uploaded_file:
    text = data_ingestion.text_extractor(uploaded_file)
    print(text)
    
    # Store text in FAISS vector database
    #vector = np.random.rand(dimension).astype('float32')  # Placeholder for actual text vectorization
    #vector_store.add_texts([text], [vector])

    #if st.button("Summarize"):
    #    summary = summarize_text(text)
    #    st.write("Summary:")
    #    st.write(summary)
        
        # Allow summary download
    #    if st.button("Download Summary"):
    #        with open("summary.txt", "w") as file:
    #            file.write(summary)
    #        st.download_button("Download the summary", data=summary, file_name="summary.txt")

