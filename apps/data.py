import sys

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import config
def app():

    st.title('Data')
    st.write("This is the `Data` page of the multi-page app.")
    st.write("The following is the DataFrame of the `iris` dataset.")
    st.subheader("Dataset")
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    if data_file is not None:
        df = pd.read_csv(data_file)
        file = pd.DataFrame(data=df)
        config.file = file
    if st.button("Process"):
        if file is not None:
            st.dataframe(file)
