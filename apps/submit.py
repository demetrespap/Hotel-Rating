import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def app():
    st.header('Submit Form')
    title = st.text_input("")
    if not title:
        st.write('Rating Review is:', title)
    else:
        st.warning("Please fill out so required fields")
    with st.form("my_form"):
        slider_val = st.slider("Rating Slider", min_value=1, max_value=5)
        checkbox_val = st.checkbox("Form checkbox")
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit Rating:")
        if submitted:
            st.write("Rating set to:", slider_val)
            st.success('This is a success message!')
