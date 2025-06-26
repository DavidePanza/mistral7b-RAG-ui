import chromadb
import streamlit as st
import fitz 
import os
from chromadb.utils import embedding_functions
from text_processing import lines_chunking, paragraphs_chunking


def get_chroma_client():
    """
    Get an ephemeral ChromaDB client for session-based RAG.
    Data is automatically deleted when user closes browser/session ends.
    """
    return chromadb.EphemeralClient()


@st.cache_resource
def initialize_chroma_client():
    """
    Initialize ChromaDB client and store in Streamlit's resource cache.
    This ensures one client per Streamlit session.
    """
    return get_chroma_client()


@st.cache_resource
def initialize_chromadb(embedding_model):
    """
    Initialize ChromaDB client and embedding function.
    Both are cached to avoid recreating on every rerun.
    """
    # Get the cached client
    client = initialize_chroma_client()

    # Initialize an embedding function (using a Sentence Transformer model)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )

    return client, embedding_func


def initialize_collection(client, embedding_func, collection_name):
    """
    Initialize a collection in ChromaDB.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def update_collection(collection, files_to_add_to_collection):
    """
    Update collection with new uploaded files.
    Returns updated collection and session state.
    """
    for file_to_add in files_to_add_to_collection:

        current_file = next(
            (file for file in st.session_state.get('uploaded_files_raw', []) 
            if file.name == file_to_add),None)
        
        if current_file is None:
            st.error(f"File '{file_to_add}' not found in uploaded files.")
            continue  

        # Read file content
        try:
            if current_file.type == "text/plain":  # Handling TXT files
                file_text = current_file.getvalue().decode("utf-8")
            elif current_file.type == "application/pdf":  # Handling PDFs
                with fitz.open(stream=current_file.getvalue(), filetype="pdf") as pdf_document:
                    file_text = "\n".join([page.get_text("text") for page in pdf_document])
            else:
                st.warning(f"Unsupported file type: {current_file.name} type:{current_file.type}")
                continue

            # Tokenize text into chunks
            max_words = 200
            chunks = lines_chunking(file_text, max_words=max_words)
            
            if not chunks:  # Skip if no chunks generated
                st.warning(f"No content extracted from {current_file.name}")
                continue

            # Store chunks in the collection
            filename = current_file.name
            collection.add(
                documents=chunks,
                ids=[f"id{filename[:-4]}.{j}" for j in range(len(chunks))],
                metadatas=[{"source": filename, "part": n} for n in range(len(chunks))],
            )
            
            st.session_state.collections_files_name.append(filename)
            st.success(f"Added {len(chunks)} chunks from {filename}")
            
        except Exception as e:
            st.error(f"Error processing {current_file.name}: {str(e)}")
            # Remove from session state if processing failed
            st.session_state.uploaded_files_name.remove(filename)
    
    return collection
    