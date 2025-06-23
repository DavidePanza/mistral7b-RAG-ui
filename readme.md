# 🚀 Streamlit RAG with Ollama Models (Local)

---

### 🎥 Demo 
![Demo GIF](https://github.com/DavidePanza/streamlit_RAG/blob/main/demo.gif)

---

## 📌 Overview
This code lets you run RAG (Retrieval-Augmented Generation) locally with Streamlit as the UI, making it easy to use. With RAG, you can upload your private documents and search for relevant information by entering prompts into an LLM (Large Language Model). This implementation is designed for users who want to build a persistent local database of relevant information that can be expanded over time by adding new documents (.txt or .pdf).


This repo combines Ollama models with ChromaDB to store and retrieve contextual information efficiently. 

What this implementation does:
- Lets you upload multiple text files to build a searchable knowledge base for the LLM.
- Retrieves relevant information from your documents using RAG with Ollama models.
- Generates AI-powered responses based on your queries.
- Helps manage database size by clearing embeddings and text chunks when needed (still a work in progress).

---

## ⚙️ Installation Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- Streamlit
- ChromaDB
- Ollama models

### Cloning the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Setting Up the Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Code
```bash
streamlit run src/run.py
```

---

## 📝 Usage
1. **Upload Files:** Drag and drop or select files to build the knowledge base.
2. **Generate Response:** Enter a custom prompt and click the 'Generate Response' button.
3. **Manage Files:** Use the dropdown menu to delete files from the database as needed.

---

## ⚠️ Important Considerations
- **Database Growth:** Deleting files only removes embeddings, not text chunks, which may increase the database size over time. To prevent excessive growth, you may need to manually delete the `chroma.sqlite3` file and recreate the database.


