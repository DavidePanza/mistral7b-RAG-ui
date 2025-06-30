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


def apply_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;700&display=swap');
    html, body, .stApp,
    .css-1v3fvcr, .css-ffhzg2, .css-1d391kg,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stText"],
    div[data-testid="stTextInput"],
    div[data-testid="stSelectbox"],
    div[data-testid="stCheckbox"],
    div[data-testid="stSlider"],
    label, input, textarea, button, select,
    .stButton, .stTextInput > div, .stMarkdown, .stCaption,
    .streamlit-expanderHeader, .st-expander > div,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Work Sans', sans-serif !important;
    }
    /* Ensure bold text uses the correct font weight */
    strong, b, .stMarkdown strong, .stMarkdown b {
        font-family: 'Work Sans', sans-serif !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)


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

    possible_paths = [
     "../images/image6.jpg",      # Local development (from src/ folder)
     "images/image6.jpg",         # Docker container (from /app)
    ]
    
    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if not image_path:
        st.error("Could not find image6.jpg in any expected location")
        return
    
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
            height: 150px;  /* Adjust the height of the background */
            width: 100%;
            margin: 0 auto;
            filter: contrast(110%) brightness(210%); /* Dim the brightness of the image */
            border-radius: 100px;  /* Makes the container's corners rounded */
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
            background-color: rgba(20, 10, 20, 0.44); /* Semi-transparent black overlay */
            z-index: 1; /* Ensure the overlay is above the image */
        }}

        /* Overlay title styling */
        .overlay-title {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: black;   /* Title color */
            font-size: 50px;
            font-weight: bold;
            text-shadow: 1px 1px 3px rgba(255, 255, 255, .0); /* Shadow for better visibility */
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
            <div class="overlay-title">Mistral-RAG</div>
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


