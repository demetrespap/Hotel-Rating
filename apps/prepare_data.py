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

# check version number
import imblearn



def get_clean(x):
  x = str(x)
  x.lower(x)
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
  str.code(code,'python')
  if str.button("Run",key=1):
    x = "hello"
    x = get_clean(x)
    str.write(x)
    print(x['Review'])
    x['Review'] = x['Review'].apply(lambda x: get_clean(x))






