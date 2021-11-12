import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def app():
    col1,cole2= st.columns(2)

    with col1:
        code = '''def hello():
            print("Algorithm !")'''
        st.code(code, 'Python')
        st.button('Run')
