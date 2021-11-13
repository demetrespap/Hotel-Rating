import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from streamlit.proto.SessionState_pb2 import SessionState


def app():
    st.header('Submit Form')
    with st.form("my_forms"):
        title = st.text_input("")
        submit = st.form_submit_button("Submission:")
        if submit:
            st.success('This is a success message!')
        slider_val = st.slider("Rating Slider", min_value=1, max_value=5, value=1)
        if slider_val:
            st.write("Rating set to:", slider_val)
    st.balloons()

