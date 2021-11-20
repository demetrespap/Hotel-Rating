import streamlit as st
from multiapp import MultiApp
from apps import home, data, prepare_data, result, submit, choose_model
import config
# import your app modules here

app = MultiApp()
html = """
  <style>

    header > .toolbar {
      flex-direction: row-reverse;
      left: 1rem;
      right: auto;
    }

    .sidebar .sidebar-collapse-control,
    .sidebar.--collapsed .sidebar-collapse-control {
      left: auto;
      right: 0.5rem;
    }

    .sidebar .sidebar-content {
      transition: margin-left .3s, box-shadow .3s;
    }

    .sidebar.--collapsed .sidebar-content {
      margin-left: auto;
      margin-right: -21rem;
    }

    .sidebar .sidebar-content {
    background-color: #111 !important;
    }

    @media (max-width: 991.98px) {
      .sidebar .sidebar-content {
        margin-left: auto;
      }
    }
  
  </style>
"""
st.markdown(html, unsafe_allow_html=True)
# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Prepare Data", prepare_data.app)
app.add_app("Choose Model", choose_model.app)
# app.add_app("Model", model.app)
app.add_app("Result", result.app)
app.add_app("Submit", submit.app)
app.run()
