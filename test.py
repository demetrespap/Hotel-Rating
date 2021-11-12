import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
image = Image.open('sunrise.jpg')

st.image(image, width=150)
st.title("Show model Data")


if st.button('Submit'):
    color = st.select_slider(
        'Select Rating Star',
        options=['0', '1', '2', '3', '4', '5'])
    st.write('Rating Star:', color)
else:
    st.write('')

# initialize list of lists
data = [['Select Option:', ''],
        ['Show Data', (11199, 11328, 11287, 32345, 12342, 1232, 13456, 123244, 13456)],
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
