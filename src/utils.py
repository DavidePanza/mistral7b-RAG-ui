import streamlit as st
import os
import sqlite3
import base64


DEFAULT_SESSION_STATE = {
    # PDF Upload
    'uploaded_files_name': [],
    'collections_files_name': [],
    'uploaded_files_raw': [],
}


def configure_page() -> None:
    """
    Configures the Streamlit page.
    """
    st.set_page_config(page_title="myRAG", 
                       layout="wide", 
                       page_icon=":rocket:")


def breaks(n=1):
    """
    Creates a line break.
    """
    if n == 1:
        st.markdown("<br>",unsafe_allow_html=True)
    elif n == 2:
        st.markdown("<br><br>",unsafe_allow_html=True)
    elif n == 3:
        st.markdown("<br><br><br>",unsafe_allow_html=True)
    else:
        st.markdown("<br><br><br><br>",unsafe_allow_html=True)


def get_base64_encoded_image(image_path):
    """
    Reads an image file and encodes it to Base64.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def load_background_image():
    """
    Loads and displays a background image with an overlaid title.
    """
    image_path = "../images/image1.jpg"  
    base64_image = get_base64_encoded_image(image_path)
    
    # Inject CSS for the background and title overlay
    st.markdown(
        f"""
        <style>
        /* Background container with image */
        .bg-container {{
            position: relative;
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: container;
            background-position: center;
            height: 250px;  /* Adjust the height of the background */
            width: 70%;
            margin: 0 auto;
            filter: contrast(110%) brightness(200%); /* Dim the brightness of the image */
            border-radius: 200px;  /* Makes the container's corners rounded */
            overflow: hidden;  
        }}

        /* Overlay for dimming effect */
        .bg-container::after {{
            content: '';
            position: absolute;
            top: ;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(20, 20, 20, 0.5); /* Semi-transparent black overlay */
            z-index: 1; /* Ensure the overlay is above the image */
        }}

        /* Overlay title styling */
        .overlay-title {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;   /* Title color */
            font-size: 70px;
            font-weight: bold;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7); /* Shadow for better visibility */
            text-align: center;
            z-index: 2; /* Ensure the title is above the overlay */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the background container with an overlaid title
    st.markdown(
        """
        <div class="bg-container">
            <div class="overlay-title">Streamlit RAG</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def initialise_session_state():
    """
    Initializes the session state variables if not already set.
    """
    for key, default_val in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


# def file_uploader() -> None:
#     """
#     Uploads multiple files.
#     """
#     st.markdown(
#     """
#     <style>
#     /* This targets the file uploader container */
#     div[data-testid="stFileUploader"] {
#         margin-top: -30px !important;
#         margin-bottom: 0px !important;
#         max-width: 450px; /* adjust as needed */
#         padding: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
#     )
#     uploaded_files = st.file_uploader(
#         "",
#         type=["txt", "pdf"], 
#         accept_multiple_files=True  
#     )
    
#     return uploaded_files


def file_uploader():
    uploaded_files = st.file_uploader(
        "",
        type=["txt", "pdf"], 
        accept_multiple_files=True)  
    
    if uploaded_files:  # Check if list is not empty
        for file in uploaded_files:  # Process each file
            if file.name not in st.session_state.uploaded_files_name:
                # Append to session state lists safely
                st.session_state.uploaded_files_name.append(file.name)
                st.session_state.uploaded_files_raw.append(file)
                st.success(f"Added new file: {file.name}")
   
    else:
        st.info("Please upload a PDF file to proceed.")


# def upload_pdf():
#     uploaded_file = st.file_uploader(
#         "",
#         type=["txt", "pdf"], 
#         accept_multiple_files=True)  
    
#     if uploaded_file is not None:
#         prev_file = st.session_state.get('uploaded_pdf_name', [])
#         if uploaded_file.name not in prev_file:
#             # New file detected
#             reset_session_state_on_upload()
#             st.session_state['pdf_changed'] = True
#         else:
#             st.session_state['pdf_changed'] = False

#         pdf_bytes = uploaded_file.read()

#         if pdf_bytes:
#             st.session_state['uploaded_pdf_bytes'] = pdf_bytes
#             st.session_state['uploaded_pdf_name'] = uploaded_file.name
#             st.success(f"File '{uploaded_file.name}' uploaded successfully!")
#         else:
#             st.error("Uploaded file is empty!")
                
#     elif uploaded_file is None and st.session_state.get('uploaded_pdf_bytes') is not None:
#         st.success("File uploaded successfully!")
#     else:
#         st.info("Please upload a PDF file to proceed.")


def reset_session_state_on_upload():
    """
    Resets session state variables to their default values.
    """
    for key, default_val in DEFAULT_SESSION_STATE.items():
        st.session_state[key] = default_val


def load_uploaded_files(uploaded_files_log):
    """
    Load the list of uploaded files from a text file.
    """
    if os.path.exists(uploaded_files_log):
        with open(uploaded_files_log, "r") as f:
            return f.read().splitlines()
    return []


def save_uploaded_files(file_list, uploaded_files_log):
    """
    Save the list of uploaded files to a text file.
    """
    with open(uploaded_files_log, "w") as f:
        f.write("\n".join(file_list))


def vacuum_db(db_path):
    """
    Run the VACUUM command on the SQLite database to optimize its performance.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM;")
    conn.commit()
    conn.close()


def remove_file_and_vectors(file_name, collection, uploaded_files_log, database_dir):
    """
    Remove a file and its associated vectors from the database.
    """
    # Remove the file from the session state
    st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f != file_name]
    
    # Save the updated list of uploaded files
    save_uploaded_files(st.session_state.uploaded_files, uploaded_files_log)
    
    # Remove the associated vectors from the database
    try:
        # Delete vectors where the metadata field "source" matches the file name
        collection.delete(where={"source": file_name})
        db_path = os.path.join(database_dir, "chroma.sqlite3")
        vacuum_db(db_path)
        st.success(f"Successfully removed {file_name} and its vectors from the database.")
    except Exception as e:
        st.error(f"Failed to remove vectors for {file_name}: {e}")


