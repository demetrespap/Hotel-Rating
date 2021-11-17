import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
from apps import choose_model
import config

def app():
    st.markdown("""
         <style>
         div.stButton > button:first-child {
             background-color: #0099ff;
             color:#ffffff;
         }
         </style>""", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>RESULT</h1>", unsafe_allow_html=True)
    st.write("""When we select the Run button, the code at the top of the screen will run. 
          This data, are the reviews that will appears in the graph shown above. This rating will be from 1 to 5 in relation to the time it took to run.  """)

    code = '''
    classification_report(y_test.astype('int'), y_pred)
    plot_confusion_matrix(nbrs, X_test, y_test.astype('int'))'''
    st.code(code, language='python')
    if st.button('Run'):
        if config.type is not None:
            st.subheader("CONFUSION MATRIX")
            plot_confusion_matrix(config.type, config.X_test, config.y_test.astype('int'))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            st.write(classification_report(config.y_test.astype('int'),config.y_pred))


