import sys

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets




def app():

    st.title('Data')
    st.write("On this page the user will need to import a csv file. This file will contain data depending on the user review and the corresponding ratings. "
             "This data will be displayed at the bottom of the screen and the csv file will be saved in the cache.")
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

