import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, rate_us # import your app modules here

app = MultiApp()

st.markdown("""
# Hotel Rating Application

This application it made for rating hotels based on their review using a Machine learning algorithm
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model", model.app)
app.add_app("Rate Us", rate_us.app)
# The main app

app.run()