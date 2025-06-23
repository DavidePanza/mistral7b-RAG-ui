import streamlit as st
import logging
import io


def configure_logging():
    """
    Configure logging.
    """
    log_stream = io.StringIO()  
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(message)s'))
    handler.setLevel(logging.WARNING) 
    
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    
    return logger, log_stream


def toggle_logging(level, logger):
    """
    Toggle logging level.
    """
    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logger.setLevel(logging.INFO)
    elif level == 'WARNING':
        logger.setLevel(logging.WARNING)
    else:
        logger.warning(f"Unknown logging level: {level}. Using WARNING as default.")
        logger.setLevel(logging.WARNING)
    for handler in logger.handlers:
        handler.setLevel(logger.level)


def display_logs(log_stream):
    """
    Display logs in the app
    """
    log_stream.seek(0)  
    logs = log_stream.read()  
    st.text(logs)  
