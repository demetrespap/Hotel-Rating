import streamlit as st
from PIL import Image


def app():
    image = Image.open('hotel.png')
    st.image(image, width=250)
    st.markdown("<h1 style='text-align: center; color: white;'>Hotel Rating Application</h1>", unsafe_allow_html=True)

    st.text("""This application it made for rating hotels based on their review using a Machine learning algorithm""")

    st.markdown("<h1 style='text-align: center; color: white;'>Contents</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: red;'>Home Page</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: red;'>Data Page</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: red;'>Choose Model Page</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: red;'>Model Page</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: red;'>Result Page</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: red;'>Submit Page</h1>", unsafe_allow_html=True)

