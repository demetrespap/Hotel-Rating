import sys
import time

import altair as alt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import config
def app():
    m = st.markdown("""
         <style>
         div.stButton > button:first-child {
             background-color: #0099ff;
             color:#ffffff;
         }
         </style>""", unsafe_allow_html=True)

    file = None
    config.file = None
    data_file = None

    st.markdown("<h1 style='text-align: center;'>UPLOAD DATA</h1>", unsafe_allow_html=True)

    st.write("On this page the user will need to import a csv file. This file will contain data depending on the user review and the corresponding ratings. "
             "This data will be displayed at the bottom of the screen and the csv file will be saved in the cache.")
    st.subheader("DATASET")
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    if data_file is not None:
        df = pd.read_csv(data_file)
        file = pd.DataFrame(data=df)
        config.file = file
    if st.button("Process"):
        if file is not None:
            st.dataframe(file)
        st.success('File Uploaded Successfully!')

    code2 ='''print(new_df['Rating'].value_counts())
    new_df.value_counts().plot.bar()'''
    st.code(code2, 'python')
    st.text_area("Printing the data",'''In this part of the code, after uploading the csv file containing the data  we print in form of plot bar the number of reviews bases on how hight the rating is (1,2,3,4,5) ''')
    if st.button('Run'):
        st.success('Code Run Successfully!')
        sns.countplot(x="Rating", data=file)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
