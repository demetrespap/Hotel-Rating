from turtle import st

import self as self
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
import config
import preprocess_kgptalkie as ps
import re
import string
from sklearn.metrics import plot_confusion_matrix
import imblearn

def get_clean(x):
  x = x.lower()
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

@str.cache
def get_clear_data(data):
  clear_data = data
  str.write(clear_data)


def app():

  #clear data
  code = '''
  def get_clean(x):
  x = x.lower()
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
        '''
  str.code(code,'python')

  #clear data function
  if str.button("Run",key=1):
    new_df = config.file
    new_df['Review'] = new_df['Review'].apply(lambda x: get_clean(x))
    #Push clean data on cache
    get_clear_data(new_df)



