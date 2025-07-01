# 🚀 Streamlit RAG Powered by Mistral 7B (4-bit) 

---

## 📌 Overview
This project implements a Retrieval-Augmented Generation (RAG) system powered by the Mistral 7B model quantized to 4-bit, hosted on Runpod. The frontend uses Streamlit for an easy-to-use UI and it is hosted on Hugging Face Spaces. This system is designed for users who want to quickly extract relevant information from their uploaded documents (.txt, .pdf).  
You can upload files here to create a temporary knowledge base that helps the AI give you relevant answers.  
**Note**: All uploaded documents and data are lost once the app is closed, ensuring your privacy and no persistent storage.

What this implementation does:
- Lets you upload multiple text files to build a searchable knowledge base for the LLM.
- Retrieves relevant information from your documents using RAG.

---
```bash
streamlit run src/run.py
```

---

## 📝 Usage
1. **Upload Files:** Drag and drop or select files to build the knowledge base.
2. **Generate Response:** Enter a custom prompt and click the 'Generate Response' button.
3. **Manage Files:** Use the dropdown menu to delete files from the database as needed.
