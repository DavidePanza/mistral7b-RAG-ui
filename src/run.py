import streamlit as st
import os
from utils import load_background_image, apply_style, configure_page, breaks, file_uploader, initialise_session_state
from mylogging import configure_logging, toggle_logging, display_logs
from collections_setup import initialize_chromadb, initialize_collection, update_collection
from runpod_setup import get_relevant_text, generate_answer, get_contextual_prompt

if __name__ == "__main__":

    configure_page()
    apply_style()
    load_background_image()
    initialise_session_state()
    breaks(2)
    st.write(
        """
    Welcome to this Streamlit app that demonstrates Retrieval-Augmented Generation (RAG) using a **Mistral-7B model hosted on Runpod** and **ChromaDB** for retrieval.

    With this app, you can:
    - Upload multiple PDF or text files to build a contextual knowledge base,
    - Ask custom questions based on your uploaded documents, and
    - Generate informed responses using a lightweight, hosted LLM.

    **Note:** All uploaded files and generated embeddings are stored **in memory only** and will be **lost when the app is closed or restarted**. No data is persisted between sessions.
    """
    )
    
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
    st.divider()

    # ---- Logging Setup ----
    use_logging = False
    if use_logging:
        logging_level = st.selectbox("Select logging level", ['INFO', 'DEBUG', 'WARNING'], index=2)
        toggle_logging(logging_level, logger)

    # ---- Vector Store Setup ----
    # Initialize ChromaDB and collection
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
    client, embedding_func = initialize_chromadb(EMBEDDING_MODEL)
    collection_name = "my_collection"
    collection = initialize_collection(client, embedding_func, collection_name)

    # Upload files
    st.markdown(
    '<h3>Upload Files</h3>',
    unsafe_allow_html=True)
    st.html("""
    Uploaded files are processed to build a contextual knowledge base for the RAG model.<br>
    When you submit a prompt, the model retrieves relevant information from these documents to generate responses.
    """)
    col1_, _, col2_ = st.columns([.4, .1, .5])
    with col1_:
        file_uploader()

    # Get the current uploaded filenames
    logger.debug(f"\n\t-- Currently uploaded files: {st.session_state.get('uploaded_files_name', 'None')}")


    # Update collection with uploaded files
    files_to_add_to_collection= [file_name for file_name in st.session_state.get("uploaded_files_name", []) if file_name not in st.session_state.get("collections_files_name", [])]
    logger.debug(f"\n\t-- Files not in collection: {files_to_add_to_collection}")

    if files_to_add_to_collection:
        collection = update_collection(collection, files_to_add_to_collection)

    # Update the session state
    logger.debug(f"Collection count: {collection.count()}")
    logger.debug(f"\n\t-- Collection data currently uploaded:")
    data_head = collection.get(limit=5)
    for i, (metadata, document) in enumerate(zip(data_head["metadatas"], data_head["documents"]), start=1):
        logger.debug(f"Item {i}:")
        logger.debug(f"Metadata: {metadata}")
        logger.debug(f"Document: {document}")
        logger.debug("-" * 40)

    # ---- Response Generation ----
    # Streamlit UI
    st.divider()
    col1, _, col2 = st.columns([.6, .01, 1])
    with col1:
        st.subheader("Enter your prompt")
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
                response = generate_answer(context_query, max_tokens=200)

            with col2:
                st.subheader("Response:")
                st.text_area("", value=response, height=200)
        else:
            logger.debug("No query provided; skipping relevant text retrieval.")
            st.warning("Please enter a prompt.")

    if use_logging:
        display_logs(log_stream)
