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

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


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
    # chart_data = pd.DataFrame(
    # np.random.randn(20, 3),
    # columns = ['a', 'b', 'c'])
    # st.line_chart(chart_data)



    # DEPENDED from choosed model
    #*******************************


    st.write("Gaussian NB")
    code3='''
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(classification_report(y_test.astype('int'), y_pred))
plot_confusion_matrix(gnb, X_test, y_test.astype('int'))
    '''
    st.code(code3, 'Python')
    st.write("KNeighborsClassifier")
    code4 = '''
nbrs = KNeighborsClassifier()
nbrs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)
print(classification_report(y_test.astype('int'), y_pred))
plot_confusion_matrix(nbrs, X_test, y_test.astype('int'))
        '''
    st.code(code4, 'Python')

    st.write("RFS")
    code5 = '''
rfs = RandomForestClassifier()
rfs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)
print(classification_report(y_test.astype('int'), y_pred))
plot_confusion_matrix(nbrs, X_test, y_test.astype('int'))
            '''
    st.code(code5, 'Python')
    st.write("CLF")
    code6 = '''
clf = LinearSVC(C=10, class_weight='balanced')
y_train=y_train.astype('int')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
            '''
    st.code(code6, 'Python')

    st.write("Get average Data ")
    # get balaned data
    code = '''def get_average_data(df):
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
    st.code(code, 'python')

    #if st.button('Run', key=2):
     #   df = data.reads_csv()
      #  print(df)
       # new_df = get_average_data(df)
        #print(new_df['Rating'].value_counts())
    st.write("NLP Represenent ")
    code7 = '''
gnb = def nlp_represent(df):
nlp = spacy.load("en_core_web_lg")
def vector_representation(row):
doc = nlp(row['Review'])
return doc.vector
df['Review_vec'] = df.apply(vector_representation, axis=1)
df_vector = df.Review_vec.apply(pd.Series)
df_vector
df_for_training = df.join(df_vector)
return df_for_training
        '''
    st.code(code7, 'Python')

    st.write("NLP Represent")

    st.write("Training collumns")
    code8 = '''
df_for_training = nlp_represent(df)
df_for_training.columns
y.value_counts().plot(kind='bar')

X = df_for_training.drop(columns=['Review', 'Rating', 'Review_vec'])
y = df_for_training['Rating']
X_train,X_test,y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state = 0)
                '''
    st.code(code8, 'Python')

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
    
