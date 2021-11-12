import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def app():
    st.header('Result')
    st.line_chart({"data": [1, 5, 2, 6, 2, 1]})
    with st.expander("See explanation"):
        st.write("""Text to describe the graph""")

