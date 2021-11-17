import numpy as np
import pandas as pd
import streamlit as st
import spacy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from streamlit.proto.SessionState_pb2 import SessionState
import preprocess_kgptalkie as ps
import re
from apps import choose_model
import config
import apps

def app():
    st.markdown("<h1 style='text-align: center;'>Submit Form</h1>", unsafe_allow_html=True)

    st.write('In this Page the user has the option to add review for the Hotel.'
             'The Algorithm takes this result and find the review. Also, in this page there is a slide bar.'
             'The value in this bar is change depends on the rating it takes.'
             'If the value is equals to 5 then balloons will appear.'
             'The code below is clean the data and takes the rating from the previous page.')

    st.subheader('Types of Prediction of given data:')
    code = '''
    nlp = spacy.load("en_core_web_lg")
    doc = nlp("very good")
    script_vector = pd.DataFrame(doc.vector)
    nbrs.predict(script_vector.T)
    
    x = 'EXCELENT'
    x = get_clean(x)
    vec = tfidf.transform([x])
    clf.predict(vec)
    '''
    st.code(code, 'python')
    st.subheader('Data converted:')

    with st.form("my_forms"):

        title = st.text_input("")
        rating = 0

        # if config.sub_pred is not None:
        submit = st.form_submit_button("Submit")
        if submit:
            if config.sub_pred == 0:
                title = get_clean(title)
                vec = config.tfidf.transform([title])
                rating = config.type.predict(vec)[0]
                rating = int(rating)
            elif config.sub_pred == 1:
                nlp = spacy.load("en_core_web_lg")
                doc = nlp(title)
                script_vector = pd.DataFrame(doc.vector)
                rating = config.type.predict(script_vector.T)[0]
                rating = int(rating)

                slider_val = st.slider("Rating Slider", min_value=1, max_value=5, value=rating)
                st.write("Rating set to:", slider_val)
            if rating == 5:
                st.success('This is a success message!')
                st.balloons()


def get_clean(x):
    x.lower()
    x.replace('\\', '')
    x.replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x