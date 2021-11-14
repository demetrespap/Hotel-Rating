import streamlit as st
from PIL import Image
from apps import result
from multiapp import MultiApp
import base64


def app():
    st.markdown("<h1 style='text-align: center;'>Hotel Rating Application</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size:15px;'>This system was created for Hotel Rating Application. It has been created by three Postgraduate Students in the CEI-523 Course, Data Science. These students are Michalis Aristotelous, Dimitris Papadopoulos and Andreas Christodoulou. Supervising professor of this course is Mr. Andreas Christoforou.This model will accept as input a file and will have the ability to calculate with great accuracy the reviews depending on the text chosen by the user and will display the corresponding number from 1 to 5.</h2>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>Contents</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size:20px;'>Home Page</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size:20px;'>Data Page</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size:20px;'>Choose Model Page</h2>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size:20px;'>Model Page</h2>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size:20px;'>Result Page</h2>",
             unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size:20px;'>Submit Page</h2>",
                unsafe_allow_html=True)
