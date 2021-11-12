import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def app():
    col1,col2= st.columns(2)

    with col1:
        code = '''def hello():
            print("Algorithm 1 !")'''
        st.code(code, 'Python')
        st.button("Run 1")
    with col2:
        code = '''def hello():
            print("Algorithm 2 !")'''
        st.code(code, 'Python')
        st.button("Run 2")

