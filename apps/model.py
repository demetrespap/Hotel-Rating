import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def app():
    code = '''def hello():
    print("Hello, Streamlit!")'''
    st.code(code, 'Python')
    st.button('Run')

