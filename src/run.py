import streamlit as st
import os
from utils import load_background_image, configure_page, breaks, file_uploader, load_uploaded_files, save_uploaded_files, remove_file_and_vectors 
from mylogging import configure_logging, toggle_logging, display_logs
from collections_setup import initialize_chromadb, initialize_collection, update_collection, get_database_directory
from src.runpod_setup import get_relevant_text, generate_answer, get_contextual_prompt

if __name__ == "__main__":

    configure_page()
    load_background_image()
    breaks(2)
    st.write(
        """
    Welcome to this Streamlit app that demonstrates how to integrate the Retrieval-Augmented Generation (RAG) 
    model with Ollama models and ChromaDB on a local machine.

    With this app, you can:
    - Upload multiple text files to build a contextual knowledge base,
    - Enter a custom prompt to generate a response, and
    - Generate a response using the RAG model.

    **Note:** This app runs only on a local machine. It implements RAG with Ollama models using Streamlit and 
    ChromaDB.  
    Be aware that when you delete files, only the embeddings are removed, while the text chunks remain.  
    Over time, this behavior can lead to an increase in the database size.  
    To prevent excessive database growth, you may need to delete the existing database and instantiate a new one.
    """
    )
    breaks(1)
    
    # # Disable Chroma telemetry
    os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
    
    # Initialize logger
    logger, log_stream = configure_logging()
    st.markdown(
        """
        <style>
        /* This targets the selectbox container */
        div[data-baseweb="select"] {
            max-width: 150px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    ) 
    logging_level = st.selectbox("Select logging level", ['INFO', 'DEBUG', 'WARNING'], index=2)
    toggle_logging(logging_level, logger)
    st.divider()
    breaks(1)

    # ---- Vector Store Setup ----
    # Initialize ChromaDB and collection
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
    client, embedding_func = initialize_chromadb(EMBEDDING_MODEL)
    collection_name = "my_collection"
    collection = initialize_collection(client, embedding_func, collection_name)

    # Define the directory for storing uploaded file names
    database_dir = get_database_directory()
    UPLOADED_FILES_LOG = os.path.join(database_dir, "uploaded_files.txt")

    # Upload files
    col1_, _, col2_ = st.columns([.4, .1, .5])
    with col1_:
        st.markdown(
            '<h3>Drag and drop or click to upload multiple files:</h3>',
            unsafe_allow_html=True
        )
        uploaded_files = file_uploader()
        breaks(1)
        st.write(
            "Uploaded files are processed to build a contextual knowledge base for the RAG model. "
            "When you submit a prompt, the model retrieves relevant information from these documents to generate more accurate and context-aware responses."
        )

    # Get the current uploaded filenames
    current_uploaded_filenames = [file.name for file in uploaded_files] if uploaded_files else []
    logger.debug(f"\n\t-- Currently uploaded files:")
    logger.debug(current_uploaded_filenames)

    # Load the previously uploaded files
    previously_uploaded_files = load_uploaded_files(UPLOADED_FILES_LOG)
    logger.debug(f"Previously uploaded files: {previously_uploaded_files}")

    # Update the session state with the new uploaded files
    st.session_state.uploaded_files = list(previously_uploaded_files)
    logger.debug(f"Updated uploaded files: {st.session_state.uploaded_files}")

    # Update collection with uploaded files
    collection, updated_session_state = update_collection(
        collection, uploaded_files, st.session_state["uploaded_files"]
    )

    # Update the session state
    st.session_state["uploaded_files"] = updated_session_state
    save_uploaded_files(st.session_state["uploaded_files"], UPLOADED_FILES_LOG)
    logger.debug(f"Collection count: {collection.count()}")
    logger.debug(f"Files in database directory: {os.listdir(get_database_directory())}")
    logger.debug(f"\n\t-- Collection data currently uploaded:")
    data_head = collection.get(limit=5)
    for i, (metadata, document) in enumerate(zip(data_head["metadatas"], data_head["documents"]), start=1):
        logger.debug(f"Item {i}:")
        logger.debug(f"Metadata: {metadata}")
        logger.debug(f"Document: {document}")
        logger.debug("-" * 40)

    # Remove files from the database
    with col2_:
        st.write("### Select Files To Remove From Database")
        available_docs = collection.count()
        if available_docs == 0:
            breaks(1)
            st.warning("No documents available in the knowledge")
        breaks(1)
        for file_name in st.session_state.uploaded_files:
            col1, col2, _ = st.columns([0.2, 0.2, 0.2])
            with col1:
                st.write(file_name)
            with col2:
                if st.button("Delete", key=f"delete_{file_name}"):
                    remove_file_and_vectors(file_name, collection, UPLOADED_FILES_LOG, database_dir)

    # ---- Response Generation ----
    # Streamlit UI
    breaks(1)
    st.divider()
    col1, _, col2 = st.columns([.6, .01, 1])
    with col1:
        st.subheader("Enter your prompt:")
        query = st.text_area("", height=200)
        generate_clicked = st.button("Generate Response")
    if generate_clicked:
        if query.strip():
            # Get the number of available documents in ChromaDB
            available_docs = collection.count()

            if available_docs > 0:
                # Ensure n_results doesn't exceed available_docs
                n_results = min(2, available_docs)
                relevant_text = get_relevant_text(collection, query=query, nresults=n_results)
            else:
                relevant_text = ""  # No documents available, so no additional context
                st.warning("No knowledge base available. Generating response based only on the prompt.")

            logger.debug("\n\t-- Relevant text retrieved:")
            logger.debug(relevant_text)

            with st.spinner("Generating response..."):
                context_query = get_contextual_prompt(query, relevant_text)
                response, _ = generate_answer(context_query, max_tokens=150)

            with col2:
                st.subheader("Response:")
                st.text_area("", value=response, height=200)
        else:
            logger.debug("No query provided; skipping relevant text retrieval.")
            st.warning("Please enter a prompt.")

    display_logs(log_stream)
