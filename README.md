# Research Paper Agent

![image](https://github.com/user-attachments/assets/421dde28-106b-4c08-bf41-82112379cb7d)

## Overview
The **Research Paper Agent** is a Generative AI tool designed to assist researchers, academics, and professionals by generating concise summaries of research papers and answering queries based on the document's content. 

The Research Paper Agent uses the **Llama 3** model to generate summaries of research papers. Users can upload a PDF document, which is processed through several text processing steps. The resulting vector embeddings are stored in a **FAISS vector store**. These embeddings can then be used to:
- Display a summary of the paper.
- Answer user queries in a Q&A format based on the content of the paper using the same LLM.

## Key Features

1. **PDF Upload and Text Extraction**:
Users can upload research papers in PDF format. The application uses `PyPDFLoader` to extract the text from the uploaded document for further processing.

2. **Text Splitting**:
The extracted text is split into smaller, manageable chunks using the `RecursiveCharacterTextSplitter`. This allows for better processing of lengthy documents by ensuring that each chunk has a reasonable size for vector embedding and summarization.

3. **Vector Embeddings**:
The text chunks are converted into vector embeddings using `OllamaEmbeddings`. These embeddings capture the semantic meaning of the text and are stored in a **FAISS vector store** for efficient retrieval.

4. **Summarization**:
The **Llama 3** model is used to generate detailed summaries of research papers. Users receive a comprehensive summary that includes details such as:
     - **Title** of the paper
     - **Authors**
     - **Abstract**
     - **Introduction**
     - **Methodology**
     - **Results**
     - **Discussion**
     - **Key Contributions**
     - **Future Work** outlined in the paper.

5. **Q&A System**:
Users can ask questions about the research paper. The LLM retrieves relevant sections from the FAISS vector store and generates an answer based on the document's content.

6. **MongoDB Integration**:
Summaries are stored in a MongoDB collection (`summary_collection`), allowing users to save and retrieve summarizations for future reference.

7. **Downloadable Summaries**:
Once a summary is generated, users can download it as a `.txt` file, enabling them to access the summary offline.

8. **Dockerization**:
The entire application has been **Dockerized** to ensure a consistent development environment and ease of deployment across different systems.


## Workflow

1. **PDF Upload**: 
Users upload a PDF file through the Streamlit interface. Once uploaded, the file is temporarily stored, and its content is extracted using `PyPDFLoader`. If a new file is uploaded, the session state is reset to avoid conflicts with previous data.

2. **Text Processing**:
The text is split into smaller chunks using the `RecursiveCharacterTextSplitter`, ensuring that the content is divided into coherent sections for further processing.

3. **Vector Embedding Creation**:
Vector embeddings for each chunk of text are generated using `OllamaEmbeddings`. These embeddings are stored in the **FAISS vector store** to allow for efficient search and retrieval later in the workflow.

4. **Summarization**:
Users can initiate the summarization process by clicking the "Summarize" button. The **Llama 3** model processes the text chunks and generates a detailed summary of the research paper, which is then stored in MongoDB.

5. **Summary Storage and Download**:
After the summarization process, users can store the summary in MongoDB by entering the paper's name. They can also download the summary as a `.txt` file for offline use.

6. **Question Answering**:
Users can interact with the uploaded research paper through the Q&A section. The system retrieves relevant sections from the FAISS vector store and provides answers based on the content of the document using the **Llama 3** model.
## Installation
To set up the Research Paper Agent locally, follow these steps:

### Clone the repository
```bash
git clone https://github.com/username/research-paper-agent.git
```

### Navigate into the project directory
```bash
cd backend
```

### Install the required dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
streamlit run app.py
```
    
