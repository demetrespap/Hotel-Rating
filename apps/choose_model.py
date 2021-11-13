import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def app():
    st.header('Choose Model')
    st.write("In the Page you have to choose the Algorithm you want to Run.")
    col1,col2= st.columns(2)

    with col1:
        code = '''def hello():
            print("Algorithm 1 !")'''
        st.code(code, 'Python')
        st.button("Run Algorithm 1")
    with col2:
        code = '''def hello():
            print("Algorithm 2 !")'''
        st.code(code, 'Python')
        st.button("Run Algorithm 2")

    #Chart
    chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns = ['a', 'b', 'c'])
    st.line_chart(chart_data)
