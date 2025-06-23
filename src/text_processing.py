import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')
nltk.download("punkt")  


def paragraphs_chunking(text, max_words=200, max_sentence_words=50):
    """
    Splits text into structured chunks, preserving paragraph integrity and avoiding unnatural breaks.
    - Uses paragraph-based splitting first.
    - Splits long paragraphs into smaller chunks based on sentence boundaries.
    """
    # Split text into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    for para in paragraphs:
        words = para.split()
        
        # If paragraph is within limit, keep as a single chunk
        if len(words) <= max_words:
            chunks.append(para)
            continue
        
        # Sentence-based chunking for large paragraphs
        sentences = sent_tokenize(para)
        chunk, chunk_word_count = [], 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence keeps chunk within word limit, add it
            if chunk_word_count + sentence_word_count <= max_words:
                chunk.append(sentence)
                chunk_word_count += sentence_word_count
            else:
                # Finalize current chunk and start a new one
                chunks.append(" ".join(chunk))
                chunk = [sentence]
                chunk_word_count = sentence_word_count

        # Append any remaining chunk
        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


def lines_chunking(text, max_words=200):
    """
    Splits text into structured chunks, preserving paragraph integrity and avoiding unnatural breaks.
    - Uses paragraph-based splitting first.
    - Splits long paragraphs into smaller chunks based on sentence boundaries.
    """
    # Split text into lines
    lines = text.splitlines()

    # Group lines into paragraphs
    paragraphs = []
    current_paragraph = []
    for line in lines:
        if line.strip():  
            current_paragraph.append(line.strip())
        else:  # Empty line indicates end of paragraph
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
    if current_paragraph: 
        paragraphs.append(" ".join(current_paragraph))

    # Process paragraphs
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
        else:
            sentences = sent_tokenize(para)
            chunk, chunk_word_count = [], 0
            for sentence in sentences:
                sentence_word_count = len(sentence.split())
                if chunk_word_count + sentence_word_count <= max_words:
                    chunk.append(sentence)
                    chunk_word_count += sentence_word_count
                else:
                    chunks.append(" ".join(chunk))
                    chunk = [sentence]
                    chunk_word_count = sentence_word_count
            if chunk:
                chunks.append(" ".join(chunk))

    return chunks