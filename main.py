import pickle
import streamlit as st
import pandas as pd
import numpy as np
from os import path

st.title("Flower Species predictor")
petal_length=st.number_input("please choose the petal length",
                             placeholder="please enter the petal length",
                             min_value=1.0, max_value=6.9,value=None)
petal_width = st.number_input("Please choose a petal width", placeholder="please enter the petal length",
                             min_value=0.1, max_value=2.5,value=None)
sepal_length = st.number_input("Please choose a sepal length",  placeholder="please enter the petal length",
                               min_value=4.3, max_value=7.9,value=None)
sepal_width = st.number_input("Please choose a sepal width", placeholder="please enter the petal length",
                              min_value=1.0, max_value=6.9,value=None)

#iris_predictor = path.join("Model","iris_classifier.pkl")
user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                          columns=['sepal_length','sepal_width','petal_length','petal_width'])
st.write(user_input)

model_path = path.join("Model","iris_classifier.pkl")
with open (model_path,'rb')as file:iris_predictor = pickle.load(file)

dict_species ={0:'setosa',1:'versicolor',2:'virginica'}

if st.button("Predict Species"):
    if ((petal_length==None) or (petal_width==None)
            or (sepal_length==None) or (sepal_width==None)):
        st.write("please fill the value")

    else:
      predicted_species = iris_predictor.predict(user_input)
      st.write("the species is", dict_species[predicted_species[0]])
