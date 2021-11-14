import sys

import altair as alt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import config
def app():

    file = None

    st.title('Data')
    st.write("On this page the user will need to import a csv file. This file will contain data depending on the user review and the corresponding ratings. "
             "This data will be displayed at the bottom of the screen and the csv file will be saved in the cache.")
    st.subheader("Dataset")
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    if data_file is not None:
        df = pd.read_csv(data_file)
        file = pd.DataFrame(data=df)
        config.file = file
    if st.button("Process"):
        if file is not None:
            st.dataframe(file)


    code2 ='''print(new_df['Rating'].value_counts())
    new_df.value_counts().plot.bar()'''
    st.code(code2, 'python')
    if st.button('Run'):

        # st.write(config.file['Rating'].value_counts().plot().bar())
        # st.write(file['Rating'].unique())
        # a = file['Rating'].value_counts()
        # sns.barplot(a)
        sns.countplot(x="Rating", data=file)
        st.pyplot()



