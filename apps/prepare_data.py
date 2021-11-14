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
    df = data.reads_csv()
    new_df = get_average_data(df)
    print(new_df['Rating'].value_counts())

  code2='''def get_average_data(df):
  max_num = (df.groupby(["Rating"]).count().sum()[0] / len(df["Rating"].unique()))
  max_num = math.trunc(max_num)
  new_df = pd.DataFrame(data = None, columns=['Review','Rating'])
  while(len(new_df.query('Rating == 1'))<=max_num or len(new_df.query('Rating == 2')) <= max_num or len(new_df.query('Rating == 3')) <= max_num
  or len(new_df.query('Rating == 4')) <= max_num or len(new_df.query('Rating == 5')) <= max_num):
    for index, row in df.iterrows():
      if (row['Rating'] == 5 and (len(new_df.query('Rating == 5')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
      if (row['Rating'] == 4 and (len(new_df.query('Rating == 4')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
      if (row['Rating'] == 3 and (len(new_df.query('Rating == 3')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
      if (row['Rating'] == 2 and (len(new_df.query('Rating == 2')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
      if (row['Rating'] == 1 and (len(new_df.query('Rating == 1')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
    print(new_df['Rating'].value_counts())
  return new_df'''
  str.code(code2,'python')

  if str.button('Run', key=2):
    x="hello"
    x=get_clean(x)
    str.write(x)
    print(new_df['Review'])
    new_df['Review'] = new_df['Review'].apply(lambda x: get_clean(x))


def get_average_data(df):
  max_num = (df.groupby(["Rating"]).count().sum()[0] / len(df["Rating"].unique()))
  max_num = math.trunc(max_num)
  new_df = pd.DataFrame(data = None, columns=['Review','Rating'])
  while(len(new_df.query('Rating == 1'))<=max_num or len(new_df.query('Rating == 2')) <= max_num or len(new_df.query('Rating == 3')) <= max_num
  or len(new_df.query('Rating == 4')) <= max_num or len(new_df.query('Rating == 5')) <= max_num):
    for index, row in df.iterrows():
      if (row['Rating'] == 5 and (len(new_df.query('Rating == 5')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
      if (row['Rating'] == 4 and (len(new_df.query('Rating == 4')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
      if (row['Rating'] == 3 and (len(new_df.query('Rating == 3')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
      if (row['Rating'] == 2 and (len(new_df.query('Rating == 2')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
      if (row['Rating'] == 1 and (len(new_df.query('Rating == 1')) <= max_num)):
        df2 = pd.DataFrame({'Review': [row['Review']], 'Rating' : [row['Rating']]})
        new_df = pd.concat([new_df, df2], ignore_index = True, axis = 0)
    print(new_df['Rating'].value_counts())
  return new_df


