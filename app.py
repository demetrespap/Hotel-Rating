import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, prepare_data, result, submit, choose_model
import config
# import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Prepare Data", prepare_data.app)
app.add_app("Choose Model", choose_model.app)
# app.add_app("Model", model.app)
app.add_app("Result", result.app)
app.add_app("Submit", submit.app)

config.tfidf = None
config.type = None
config.X_test = None
config.y_test = None
config.y_pred = None
config.sub_pred = None
# The main app
app.run()
