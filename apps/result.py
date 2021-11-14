import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



def app():
    st.header('Result')
    code = '''
    print(classification_report(y_test.astype('int'), y_pred))
    plot_confusion_matrix(nbrs, X_test, y_test.astype('int'))'''
    st.code(code, language='python')
    st.button('Run')


    st.line_chart({"data": [1, 5, 2, 5, 2, 1]})
    with st.expander("See explanation for the above:"):
        st.write("""When we select the Run button, the code at the top of the screen will run. 
        This data, are the reviews that will appears in the graph shown above. This rating will be from 1 to 5 in relation to the time it took to run.  """)




