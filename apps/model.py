import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def app():
    m = st.markdown("""
         <style>
         div.stButton > button:first-child {
             background-color: #0099ff;
             color:#ffffff;
         }
         </style>""", unsafe_allow_html=True)
    st.header('RUN MODEL')
    st.write("In this Page user is select the Model that they want to Run.")
    code = '''def hello():
    print("Hello, Streamlit!")'''
    st.code(code, 'Python')
    st.button("Run")
