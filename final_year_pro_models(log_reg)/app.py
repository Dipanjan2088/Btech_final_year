import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle
import numpy
import os

# load pre-trained model
loaded_model1 = pickle.load(open('pca_final.pk1' , 'rb'))
loaded_model2 = pickle.load(open('final_model.pk1' , 'rb'))

uploaded_file = st.file_uploader("Upload your file here...", type=['csv'])

if uploaded_file is not None:
    df100=pd.read_csv(uploaded_file)
    result1=loaded_model1.fit_transform(df100)
    result=loaded_model2.predict(result1)
    count=1
    for i in result:
      if i>0.5:
        st.write(count,"th person is churn")
      else:
        st.write(count,"th person is not churn")
      count=count+1
