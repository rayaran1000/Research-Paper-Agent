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

# LLM Loading
# Loading the model(Llama3 model)
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-8b-8192")


# Chat prompt template(For summarization)
prompt=ChatPromptTemplate.from_template([
    ("System","You are a highly capable AI language model, skilled at understanding and summarizing complex research papers. You can extract key information, identify main ideas, and provide concise summaries."),
    ("Human",
    """I have uploaded the textual content of a research paper. Please summarize the paper, including the following details:
    Title: Provide the title of the paper.
    Authors: List the authors of the paper.
    Abstract: Summarize the abstract of the paper.
    Introduction: Provide a brief summary of the introduction.
    Methods: Summarize the methodology used in the paper.
    Results: Outline the main findings or results of the study.
    Discussion: Summarize the key points from the discussion section.
    Conclusion: Provide the conclusion of the paper.
    Key Contributions: Highlight the key contributions of the paper.
    Future Work: Mention any future work or recommendations given by the authors.
    The textual content of the research paper is here : {input}
    """)
])

# Document Retreival chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Document Retreival chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit app
st.title("Research Paper Summarizer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

data_ingestion = DataIngestion()

# Creating a session state in streamlit(Happens only when we want to store)
if st.button('Store'):
    if "vectors" not in st.session_state:
        # Setting the embeddings and getting the doc from webpage
        st.session_state.embeddings=HuggingFaceBgeEmbeddings()
        if uploaded_file:
            st.session_state.docs=data_ingestion.text_extractor(uploaded_file)
        
        # Chunking the data and storing in vector database after text split
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
        st.session_state.vectors=FAISS.from_documents(index,st.session_state.final_documents,st.session_state.embeddings)

# Summarization 
if st.button("Summarize"):
    summary = retrieval_chain.invoke({"input":st.session_state.docs})
    st.write("Summary:")
    st.write(summary)
        
# Allow summary download
    if st.button("Download Summary"):
        with open("summary.txt", "w") as file:
            file.write(summary)
            st.download_button("Download the summary", data=summary, file_name="summary.txt")

