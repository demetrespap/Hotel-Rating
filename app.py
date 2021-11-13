import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, rate_us, result, submit, prepare_data

# import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Prepare Data", prepare_data.app)
app.add_app("Model", model.app)
app.add_app("Choose Model", rate_us.app)
app.add_app("Result", result.app)
app.add_app("Submit", submit.app)

# The main app
app.run()
