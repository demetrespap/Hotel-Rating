from turtle import st
import streamlit as str
import spacy
import pandas as pd
import numpy as np
import seaborn as sns
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from apps import data
import preprocess_kgptalkie as ps
import re
from sklearn.metrics import plot_confusion_matrix
import config
from apps import data
# check version number
import imblearn



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

def app():
  str.markdown("""
       <style>
       div.stButton > button:first-child {
           background-color: #0099ff;
           color:#ffffff;
       }
   
       </style>""", unsafe_allow_html=True)
  str.markdown("<h1 style='text-align: center;'>PREPARE DATA</h1>", unsafe_allow_html=True)

  str.write("In this Page we clean the Data.We will capture the data from the csv file, from the previous page.")

  code = '''
  def get_clean(x):
  x = str(x).lower().replace('\\', '').replace('_', ' ')
  x = ps.cont_exp(x)
  x = ps.remove_emails(x)
  x = ps.remove_urls(x)
  x = ps.remove_html_tags(x)
  x = ps.remove_rt(x)
  x = ps.remove_accented_chars(x)
  x = ps.remove_special_chars(x)
  x = re.sub("(.)\\1{2,}", "\\1", x)
  return x
        '''
  str.code(code, 'python')
  if str.button("Run",key=1):
    x = config.file
    str.write(x)
    x['Review'] = x['Review'].apply(lambda x: get_clean(x))
    print(x['Review'])
    str.success("the process succeded")

  str.write(
    "•The lower () method takes no arguments and returns the lowercased strings from the given string by converting each uppercase character to lowercase. "
    "If there are no uppercase characters in the given string, it returns the original string.This code appears in the first line of the above code"
    ":x = str(x).lower().replace('\\', '').replace('_', ' ').")
  str.write(
    "•Compound statements span multiple lines, although in simple incarnations a whole compound statement may be contained in one line. This code appears in the second line:x =ps.cont_exp(x).")
  str.write(
    "•Remove emails expression will delete if there is emails in the code. This code appears in the third line: x = ps.remove_emails(x).")
  str.write(
    "•Remove URLs expression will delete if there is URLs in the code. This code appears in the third line: x = ps.remove_urls(x).")
  str.write(
    "•Remove Html Tags expression will delete if there is Html Tags in the code. This code appears in the third line: x = ps.remove_html_tags(x).")
  str.write(
    "•Remove Strings expression will delete if there is Strings in the object added to filename. This code appears in the third line:   x = ps.remove_rt(x).")
  str.write(
    "•Remove Accented Characters expression will delete Unidecode and convert it to Ascii Code. This code appears in the third line:   x = ps.remove_accented_chars(x).")
  str.write(
    "•Remove Special Characters expression will delete all special characters in the code. This code appears in the third line:     x = ps.remove_special_chars(x).")
  str.write("•This Regular Expression x = re.sub('(.)\\1{2,}', '\\1', x) is used to normalize the data.")
