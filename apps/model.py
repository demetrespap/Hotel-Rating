import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def app():
    st.header('Run Model')
    st.write("In this Page user is select the Model that they want to Run.")
    code = '''def hello():
    print("Hello, Streamlit!")'''
    st.code(code, 'Python')
    st.button("Run")



