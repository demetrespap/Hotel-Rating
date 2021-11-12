import streamlit as st

def app():
    st.title('Home')

    st.write('Home Page')

    st.write('This is page 1')
    label="Upload your training data here:"
    st.sidebar.file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None)