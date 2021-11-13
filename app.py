import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, choose_model, result, submit

# import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model", model.app)
app.add_app("Choose Model", choose_model.app)
app.add_app("Result", result.app)
app.add_app("Submit", submit.app)

# The main app
app.run()
