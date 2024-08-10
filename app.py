import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    st.error("Groq API key is missing. Please set the GROQ_API_KEY environment variable.")
    st.stop()

langchain_api_key = os.environ.get('LANGCHAIN_API_KEY')
if not langchain_api_key:
    st.error("LangChain API key is missing. Please set the LANGCHAIN_API_KEY environment variable.")
    st.stop()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Load the LLM (Llama3 model)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Chat prompt template for summarization
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly capable AI language model, skilled at understanding and summarizing complex research papers. You can extract key information, identify main ideas, and provide concise summaries."),
    ("human",
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
    The textual content of the research paper is here: {context}
    """)
])

# Chat prompt template for question answering
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly capable AI language model, skilled at answering questions based on the content of research papers."),
    ("human", "Here is the content of the research paper: {context}. Now, I have a question: {question}")
])

# Streamlit app
st.title("Research Paper Summarizer and QA System")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Reset the session state when a new file is uploaded
if uploaded_file is not None:
    if "uploaded_filename" in st.session_state and st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.embeddings = None
        st.session_state.loader = None
        st.session_state.docs = None
        st.session_state.text_splitter = None
        st.session_state.final_documents = None
        st.session_state.vectors = None

    st.session_state.uploaded_filename = uploaded_file.name

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load PDF and extract content
        st.session_state.loader = PyPDFLoader(temp_file_path)
        st.session_state.docs = st.session_state.loader.load()
        if not st.session_state.docs:
            st.error("Failed to extract text from the PDF. Please ensure the file is not empty or corrupted.")
            st.stop()

        # Text splitting and vector storage
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

        st.success("PDF uploaded and processed successfully.")

    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        st.stop()

    # Summarization chain setup
    document_chain = create_stuff_documents_chain(llm, summary_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Summarization
    if st.button("Summarize"):
        with st.spinner("Summarizing the document..."):
            try:
                summary = retrieval_chain.invoke({"context": st.session_state.docs, "input": "Give me summary"})
                if not summary:
                    st.error("The summarization process failed. Please try again.")
                else:
                    st.write("Summary:")
                    st.write(summary)
                    st.session_state.summary = summary
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")

    # Allow summary download
    if "summary" in st.session_state and st.session_state.summary:
        if st.button("Download Summary"):
            try:
                with open("summary.txt", "w") as file:
                    file.write(st.session_state.summary)
                st.download_button("Download the summary", data=st.session_state.summary, file_name="summary.txt")
            except Exception as e:
                st.error(f"Failed to create the summary file: {e}")

    # Question Answering
    st.write("### Question Answering")
    question = st.text_input("Enter your question:")
    if question and st.button("Get Answer"):
        with st.spinner("Searching for the answer..."):
            try:
                qa_chain = create_stuff_documents_chain(llm, qa_prompt)
                qa_chain_response = qa_chain.invoke({"context": st.session_state.docs, "question": question})
                if not qa_chain_response:
                    st.error("Failed to retrieve an answer. Please try a different question.")
                else:
                    st.write("Answer:")
                    st.write(qa_chain_response)
            except Exception as e:
                st.error(f"An error occurred during the question answering process: {e}")
