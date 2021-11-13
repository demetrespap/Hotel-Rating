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

    file=reads_csv()
    print(file)
    if st.button("Process"):
        if file is not None:

            file_details = {"Filename": file.name, "FileType": file.type, "FileSize": file.size}
            st.write(file_details)

            st.dataframe(file)

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def reads_csv():
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    df = pd.read_csv(data_file)
    return df

