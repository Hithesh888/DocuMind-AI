

\# DocuMind AI — RAG Chatbot for Documents



A document question-answering chatbot built using Retrieval-Augmented Generation (RAG), allowing users to ask questions about uploaded documents and receive context-based answers with source references.



\## Features



\- Upload PDF, TXT, and CSV documents

\- Extract and split text into chunks

\- Generate embeddings using Sentence Transformers

\- Store and search document chunks with ChromaDB

\- Generate answers locally using Ollama and Llama 3.2 1B

\- Display source references for answers

\- Interactive Streamlit web interface



\## Technology Stack



\- Python

\- Streamlit

\- ChromaDB

\- Sentence Transformers

\- Ollama

\- Llama 3.2 1B

\- PyPDF



\## How It Works



1\. Upload a document.

2\. Extract its text and divide it into chunks.

3\. Convert the chunks into embeddings.

4\. Store the embeddings in ChromaDB.

5\. Retrieve relevant chunks for a user's question.

6\. Generate an answer using the local language model.

7\. Display the answer and its source references.



\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/YOUR-USERNAME/DocuMind-AI.git

cd DocuMind-AI

```



Replace `YOUR-USERNAME` with your GitHub username.



\### 2. Create and activate a virtual environment



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```powershell

python -m pip install -r requirements.txt

```



\### 4. Install Ollama



Download Ollama from https://ollama.com/download.



Then download the model:



```bash

ollama pull llama3.2:1b

```



\### 5. Run the application



```bash

python -m streamlit run app.py

```



Open the local URL shown in your terminal, usually http://localhost:8501.



\## Privacy



Uploaded documents and the local ChromaDB database are not intended to be published in this repository. The `.gitignore` file excludes local data and environment files.



\## Future Improvements



\- Improve retrieval accuracy

\- Add document management features

\- Evaluate answer quality

\- Add automated tests



\## Author



\*\*Hithesh J\*\*



GitHub: https://github.com/Hithesh888



