import sys

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets




def app():

    st.title('Data')
    st.write("This is the `Data` page of the multi-page app.")
    st.write("The following is the DataFrame of the `iris` dataset.")
    st.subheader("Dataset")
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    file=read_csv(data_file)
    print(data_file)
    if st.button("Process"):
        if data_file is not None:

            file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
            st.write(file_details)

            st.dataframe(file)

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def read_csv(f):
    df = pd.read_csv(f)
    return df

