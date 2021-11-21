import numpy as np
import pandas as pd
import streamlit as st
import spacy
from altair.vegalite.v3.theme import theme
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
    st.markdown("""
         <style>
         div.stButton > button:first-child {
             background-color: #0099ff;
             color:#ffffff;
         }
         </style>""", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>SUBMIT FORM</h1>", unsafe_allow_html=True)

    st.write('In the last part of the process, the user has the option to add review for the Hotel.'
             ' The Algorithm takes this result and find the review after clicking submit.'
             ' The value of the slide bar ate the bottom changes according the rating it is given'
             ' If the value is equals to 5 then balloons will appear.'
             ' The code below is clean the data and takes the rating from the previous page.')

    st.subheader('TYPES OF PREDICTION OF GIVEN DATA:')
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
    st.subheader('DATA CONVERTED:')

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
                st.success('Review added successful')
                st.balloons()


def get_clean(x):
    x.lower()
    x.replace('\\', '')
    x.replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x