import chromadb
import streamlit as st
import fitz 
import os
from chromadb.utils import embedding_functions
from text_processing import lines_chunking, paragraphs_chunking


# def get_database_directory():
#     """
#     Get the directory for storing the database.
#     """
#     # Use an absolute path for better reliability
#     parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#     persist_dir = os.path.join(parent_dir, "database")

#     # Create directory if it doesn't exist
#     os.makedirs(persist_dir, exist_ok=True)
    
#     return persist_dir


# def get_chroma_client():
#     """
#     Get a ChromaDB client.
#     """
#     persist_dir = get_database_directory()
#     return chromadb.PersistentClient(path=persist_dir)


# def initialize_chromadb(EMBEDDING_MODEL):
#     """
#     Initialize ChromaDB client and embedding function.
#     """
#     # Create a persistent directory for storing the database
#     client = get_chroma_client()

#     # Initialize an embedding function (using a Sentence Transformer model)
#     embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name=EMBEDDING_MODEL
#     )

#     return client, embedding_func


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
    

# def update_collection(collection, uploaded_files, session_state):
#     """
#     Update collection with new uploaded files.
#     Returns updated collection and session state.
#     """
#     # Convert session_state to a set for efficient lookups
#     session_state_set = set(session_state) if session_state else set()
    
#     for uploaded_file in uploaded_files:
#         if uploaded_file.name not in session_state_set:
#             # Add the file name to the session state set
#             session_state_set.add(uploaded_file.name)
            
#             # Read file content
#             try:
#                 if uploaded_file.type == "text/plain":  # Handling TXT files
#                     file_text = uploaded_file.getvalue().decode("utf-8")
#                 elif uploaded_file.type == "application/pdf":  # Handling PDFs
#                     pdf_document = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
#                     file_text = "\n".join([page.get_text("text") for page in pdf_document])
#                     pdf_document.close()  # Close the PDF document
#                 else:
#                     st.warning(f"Unsupported file type: {uploaded_file.type}")
#                     continue

#                 # Tokenize text into chunks
#                 max_words = 200
#                 chunks = lines_chunking(file_text, max_words=max_words)
                
#                 if not chunks:  # Skip if no chunks generated
#                     st.warning(f"No content extracted from {uploaded_file.name}")
#                     continue

#                 # Store chunks in the collection
#                 filename = uploaded_file.name
#                 collection.add(
#                     documents=chunks,
#                     ids=[f"id{filename[:-4]}.{j}" for j in range(len(chunks))],
#                     metadatas=[{"source": filename, "part": n} for n in range(len(chunks))],
#                 )
                
#                 st.success(f"Added {len(chunks)} chunks from {filename}")
                
#             except Exception as e:
#                 st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#                 # Remove from session state if processing failed
#                 session_state_set.discard(uploaded_file.name)
    
#     # Convert the set back to a list for session state
#     updated_session_state = list(session_state_set)
    
#     return collection, updated_session_state


# def update_collection(collection, uploaded_files, session_state):
#     # Convert session_state to a set for efficient lookups
#     session_state_set = set(session_state)
    
#     for uploaded_file in uploaded_files:
#         if uploaded_file.name not in session_state_set:
#             # Add the file name to the session state set
#             session_state_set.add(uploaded_file.name)
            
#             # Read file content
#             if uploaded_file.type == "text/plain":  # Handling TXT files
#                 file_text = uploaded_file.getvalue().decode("utf-8")
#             elif uploaded_file.type == "application/pdf":  # Handling PDFs
#                 pdf_document = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
#                 file_text = "\n".join([page.get_text("text") for page in pdf_document])
#             else:
#                 file_text = ""

#             # Tokenize text into chunks
#             max_words = 200
#             chunks = lines_chunking(file_text, max_words=max_words)

#             # Store chunks in the collection
#             filename = uploaded_file.name
#             collection.add(
#                 documents=chunks,
#                 ids=[f"id{filename[:-4]}.{j}" for j in range(len(chunks))],
#                 metadatas=[{"source": filename, "part": n} for n in range(len(chunks))],
#             )
    
#     # Convert the set back to a list for session state
#     updated_session_state = list(session_state_set)
    
    # return collection, updated_session_state