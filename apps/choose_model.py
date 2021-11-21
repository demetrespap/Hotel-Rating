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
from apps import prepare_data
import re
import string
from sklearn.metrics import plot_confusion_matrix
import imblearn
import config
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def app():
    m = st.markdown("""
         <style>
         div.stButton > button:first-child {
             background-color: #0099ff;
             color:#ffffff;
         }
         </style>""", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>CHOOSE MODEL</h1>", unsafe_allow_html=True)
    st.write("In this screen you can choose between 4 different algorithms in order to start the learning process. Also we explain some functions which manipulate the dataset in order for the algorithm to receive correct and valid data")

    col1,col2 = st.columns(2)
    col3,col4 = st.columns(2)

    with col1:
        st.write("**Gaussian NB**")
        code3 = '''
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(classification_report(y_test.astype('int'), y_pred))
plot_confusion_matrix(gnb, X_test, y_test.astype('int'))
            '''
        st.code(code3, 'Python')

    with col2:
        st.write("**KNeighborsClassifier**")
        code4 = '''
nbrs = KNeighborsClassifier()
nbrs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)
print(classification_report(y_test.astype('int'), y_pred))
plot_confusion_matrix(nbrs, X_test, y_test.astype('int'))
                '''
        st.code(code4, 'Python')

    with col3:
        st.write("**RFS**")
        code5 = '''
rfs = RandomForestClassifier()
rfs.fit(X_train, y_train)
y_pred = nbrs.predict(X_test)
print(classification_report(y_test.astype('int'), y_pred))
plot_confusion_matrix(nbrs, X_test, y_test.astype('int'))
                    '''
        st.code(code5, 'Python')

    with col4:
        st.write("**CLF**")
        code6 = '''
clf = LinearSVC(C=10, class_weight='balanced')
y_train=y_train.astype('int')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
                 '''
        st.code(code6, 'Python')


    st.title("Algorithms featured")
    st.text_area("Gaussian NB",'''Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. The Gaussian Naive Bayes algorithm is a special type of NB algorithm.
It's specifically used when the features have continuous values. It's also assumed that all the features are following a gaussian distribution ''',height=150)
    st.text_area("KNeighborsClassifier",''' K-Nearest Neighbors, or KNN for short, is one of the simplest machine learning algorithms and is used in a wide array of institutions. KNN is a non-parametric, lazy learning algorithm. When we say a technique is non-parametric, it means that it does not make any assumptions about the underlying data. In other words, it makes its selection based off of the proximity to other data points regardless of what feature the numerical values represent. Being a lazy learning algorithm implies that there is little to no training phase. Therefore, we can immediately classify new data points as they present themselves. ''',height=150)
    st.text_area("RFC (Random Forest Classification)",''' Random forest is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result. In simple words random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction. ''',height=150)
    st.text_area("CLF",'''  Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.
This class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme. ''',height=150)
    algo=st.selectbox("Choose an algorithm to run",(' ','Gaussian NB','KNeighborsClassifier','RFS','CLF'))
    st.write(config.file)
    if algo is not ' ' and algo is not None:
        if config.file is not None:
            file = config.file
            if algo == 'Gaussian NB':
                st.info("Gaussian NB algorithm started")
                df_for_training = nlp_represent(file)
                X = df_for_training.drop(columns=['Review', 'Rating', 'Review_vec'])
                y = df_for_training['Rating']
                gaussianNB(X,y)


            elif algo == 'KNeighborsClassifier':
                    st.info("KNeighborsClassifier algorithm starting")
                    df_for_training = nlp_represent(file)
                    X = df_for_training.drop(columns=['Review', 'Rating', 'Review_vec'])
                    y = df_for_training['Rating']
                    KNeighboursClassifier(X,y)


            elif algo == 'RFS':
                    st.info("RFS algorithm starting")
                    df_for_training = nlp_represent(file)
                    X = df_for_training.drop(columns=['Review', 'Rating', 'Review_vec'])
                    y = df_for_training['Rating']
                    RandomClassifiers(X,y)


            elif algo == 'CLF':
                    st.info("CLF algorithm starting")
                    df_for_training = get_average_data(file)
                    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 5), analyzer='char')
                    config.tfidf = tfidf
                    X = tfidf.fit_transform(df_for_training['Review'])
                    y = df_for_training['Rating']
                    clf(X,y)



    #DEPENDED from choosed model
    #*******************************
    st.title("GET AVERAGE DATA: ")
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
    st.text_area("",'''This part of the code it is summoned only for the CLF algorithm before it starts running and it is summoned in order to check for low rating values because they are much less tha high rating values and the duplicates the values. This is done in order for the algorith to hava an equale number of values in all rating scale.''', height=150)

# NLP representer
# -------------------------------------------------------------------------
    st.title("NLP Represenent ")
    code7 = '''
def nlp_represent(df):
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
    st.text_area("",''' It is summoned by the gaussianNB, KneighborClassifier and RFS algorithm. This part of the code is responsible for  ''',height=150)

#Training collumns
#-------------------------------------------------------------------------
    st.title("Training collumns")
    code8 = '''
df_for_training = nlp_represent(df)
X = df_for_training.drop(columns=['Review', 'Rating', 'Review_vec'])
y = df_for_training['Rating']
X_train,X_test,y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state = 0)
                '''
    st.code(code8, 'Python')
    st.text_area(" ",''' In this areas the txt is transformes in to numbers and we split the data with 75% of the going for the testing and the rest 25% going for the model training ''')

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

def nlp_represent(df):
    encore="en_core_web_lg"
    nlp = spacy.load(encore)
    def vector_representation(row):
        doc = nlp(row['Review'])
        return doc.vector
    df['Review_vec'] = df.apply(vector_representation, axis=1)
    df_vector = df.Review_vec.apply(pd.Series)
    df_vector
    df_for_training = df.join(df_vector)
    return df_for_training


# ALGORITHMS
# ===============================================================

def gaussianNB(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    config.type = gnb
    config.X_test = X_test
    config.y_test = y_test
    config.y_pred = y_pred
    config.sub_pred = 1
    st.success("The algorithm completed")

def KNeighboursClassifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    nbrs = KNeighborsClassifier()
    nbrs.fit(X_train, y_train)
    y_pred = nbrs.predict(X_test)
    config.type = nbrs
    config.X_test = X_test
    config.y_test = y_test
    config.y_pred = y_pred
    config.sub_pred = 1
    st.success("The algorithm completed")

def RandomClassifiers(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    rfs = RandomForestClassifier()
    rfs.fit(X_train, y_train)
    y_pred = rfs.predict(X_test)
    config.type = rfs
    config.X_test = X_test
    config.y_test = y_test
    config.y_pred = y_pred
    config.sub_pred = 1
    st.success("The algorithm completed")


def clf(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = LinearSVC(C=10, class_weight='balanced')
    y_train = y_train.astype('int')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    config.type = clf
    config.X_test = X_test
    config.y_test = y_test
    config.y_pred = y_pred
    config.sub_pred = 0
    st.success("The algorithm completed")
