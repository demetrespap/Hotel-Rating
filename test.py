import numpy as np
import pandas as pd
import streamlit as st

st.title("Show model Data")

if st.button('Say hello'):
    st.write('test')
else:
    st.write('Goodbye')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# initialize list of lists
data = [['Show Data', (11199, 11328, 11287, 32345, 12342, 1232, 13456, 123244, 13456)],
        ['Clean Data', ('a', 'b', 'c')],
        ['Train Model', ("loc2", "loc1", "loc3", "loc1", "loc2", "loc2", "loc3", "loc2", "loc1")],
        ['Show Graphs', ("loc2", "loc1", "loc3", "loc1", "loc2", "loc2", "loc3", "loc2", "loc1")]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Name', 'ID'])

values = df['Name'].tolist()
options = df['ID'].tolist()
dic = dict(zip(options, values))

a = st.sidebar.selectbox('Choose Data', options, format_func=lambda x: dic[x])

st.write(a)