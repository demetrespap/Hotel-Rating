import self as self
import streamlit as st
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
import string
from sklearn.metrics import plot_confusion_matrix
import imblearn


def app():
    st.header('Choose Model')
    st.write("In the Page you have to choose the Algorithm you want to Run.")
    col1,col2= st.columns(2)

    with col1:
        code = '''def hello():
            print("Algorithm 1 !")'''
        st.code(code, 'Python')
        st.button("Run Algorithm 1")
    with col2:
        code = '''def hello():
            print("Algorithm 2 !")'''
        st.code(code, 'Python')
        st.button("Run Algorithm 2")

    #Chart
    chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns = ['a', 'b', 'c'])
    st.line_chart(chart_data)



    # DEPENDED from choosed model
    #*******************************

    # get balaned data
    code2 = '''def get_average_data(df):
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
    st.code(code2, 'python')

    if st.button('Run', key=2):
        df = data.reads_csv()
        print(df)
        new_df = get_average_data(df)
        print(new_df['Rating'].value_counts())


def get_average_data(df):
    max_num = (df.groupby(["Rating"]).count().sum()[0] / len(df["Rating"].unique()))
    max_num = math.trunc(max_num)
    new_df = pd.DataFrame(data=None, columns=['Review', 'Rating'])
    while (len(new_df.query('Rating == 1')) <= max_num or len(new_df.query('Rating == 2')) <= max_num or len(
            new_df.query('Rating == 3')) <= max_num
           or len(new_df.query('Rating == 4')) <= max_num or len(new_df.query('Rating == 5')) <= max_num):
        for index, row in df.iterrows():
            if (row['Rating'] == 5 and (len(new_df.query('Rating == 5')) <= max_num)):
                df2 = pd.DataFrame({'Review': [row['Review']], 'Rating': [row['Rating']]})
                new_df = pd.concat([new_df, df2], ignore_index=True, axis=0)
            if (row['Rating'] == 4 and (len(new_df.query('Rating == 4')) <= max_num)):
                df2 = pd.DataFrame({'Review': [row['Review']], 'Rating': [row['Rating']]})
                new_df = pd.concat([new_df, df2], ignore_index=True, axis=0)
            if (row['Rating'] == 3 and (len(new_df.query('Rating == 3')) <= max_num)):
                df2 = pd.DataFrame({'Review': [row['Review']], 'Rating': [row['Rating']]})
                new_df = pd.concat([new_df, df2], ignore_index=True, axis=0)
            if (row['Rating'] == 2 and (len(new_df.query('Rating == 2')) <= max_num)):
                df2 = pd.DataFrame({'Review': [row['Review']], 'Rating': [row['Rating']]})
                new_df = pd.concat([new_df, df2], ignore_index=True, axis=0)
            if (row['Rating'] == 1 and (len(new_df.query('Rating == 1')) <= max_num)):
                df2 = pd.DataFrame({'Review': [row['Review']], 'Rating': [row['Rating']]})
                new_df = pd.concat([new_df, df2], ignore_index=True, axis=0)
        print(new_df['Rating'].value_counts())
    return new_df
    
