from re import M
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# import tensorflow as tf


st.set_page_config(page_title="Fraud Detection")

st.markdown("<h1 style='text-align: center; color: Indigo;' Fraud Detection Application</h1>", unsafe_allow_html=True)

_, col2, _ = st.columns([2.3, 3, 2])

with col2:  
    # img = Image.open("churn.jpg")
    # st.image(img, width=320)

    img = Image.open("img1.png")
    st.image(img)

st.success('###### FRAUD DETECTION')


st.info("###### Feature Information\n"
         "###### Time: This feature is contains the seconds elapsed between each transaction and the first transaction in the dataset.\n"
         "###### Amount: This feature is the transaction Amount, can be used for example-dependant cost-senstive learning.\n"
         "###### Class: This feature is the target variable and it takes value 1 in case of fraud and 0 otherwise.\n")
        

import pickle
# from tensorflow.keras.models import load_model


col, col2 = st.columns([4, 4])
with col:
    st.markdown("###### Please ... ")

    Time = st.number_input("Time", min_value=0, max_value=172792,  value=406)


    
with col2:
    st.markdown("###### Please select an Amount")

    Amount = st.number_input("Amount", min_value=0, max_value=25691,  value=0)



col1, col2, col3 = st.columns([10, 10, 10])

with col1:
    V1 = st.number_input("V1", min_value=-56.0, max_value=2.4, value=-0.5)
    V2 = st.number_input("V2", min_value=-72.0, max_value=22.0  ,value=2.2)
    V3 = st.number_input("V3", min_value=-48.0, max_value=9.0 ,value=-3.5)
    V4 = st.number_input("V4", min_value=-5.0, max_value=16.0  ,value=0.2)
    V5 = st.number_input("V5", min_value=-113.0, max_value=34.8 ,   value=1.9)
    V6 = st.number_input("V6", min_value=-26.0, max_value=73.0  , value=-1.2)
    V7 = st.number_input("V7 ",min_value=-43.0, max_value=120.5,  value=-0.3)
    V8 = st.number_input("V8 ",min_value=-73.0, max_value=20.0,  value=0.5)
    V9 = st.number_input("V9 ",min_value=-13.0, max_value=15.59,  value=-1.8)
    V10 =st.number_input("V10",min_value=-24.0, max_value=27.75,  value=-3.4)

with col2:
    V11 =st.number_input("V11",min_value=-4.0, max_value=12.0, value=2.2)
    V12 =st.number_input("V12",min_value=-18.0, max_value=7.84,  value=-1.7)
    V13 = st.number_input("V13",min_value=-5.0, max_value=7.1,  value=0.5)
    V14 = st.number_input("V14", min_value=-19.0, max_value=10.52,  value=-4.4)
    V15 = st.number_input("V15", min_value=-4.0, max_value=8.87,  value=-1.2)
    V16 = st.number_input("V16", min_value=-14.0, max_value=17.31,  value=-1.2)
    V17 = st.number_input("V17", min_value=-25.0, max_value=9.25,  value=-1.8)
    V18 = st.number_input("V18", min_value=-9.0, max_value=5.0,  value=-0.1)
    V19 =st.number_input("V19", min_value=-7.0, max_value=5.59,  value=0.4)
    V20 =st.number_input("V20", min_value=-54.0, max_value=39.42,  value=0.1)

with col3:
    
    V21 =st.number_input("V21", min_value=-34.0, max_value=27.20,  value=0.3)
    V22 =st.number_input("V22", min_value=-10.0, max_value=10.5,  value=0.2)
    V23 =st.number_input("V23", min_value=-44.0, max_value=22.52,  value=-0.3)
    V24 =st.number_input("V24", min_value=-2.0, max_value=4.58,  value=0.1)
    V25 = st.number_input("V25 ", min_value=-10.0, max_value=7.51,  value=0.3)
    V26 = st.number_input("V26 ", min_value=-2.0, max_value=3.51,  value=0.6)
    V27 = st.number_input("V27 ", min_value=-22.0, max_value=31.61,  value=0.2)
    V28 = st.number_input("V28", min_value=-15.0, max_value=33.84,  value=0.1)
   


my_dict = {'Time': Time, 'V1':V1, 'V2':V2, 'V3': V3,'V4':V4,'V5':V5,'V6': V6, 'V7':V6,'V8': V8, 'V9':V9,'V10':V10,
'V11': V1, 'V12': V12,'V13': V13, 'V14': V14,'V15': V15,'V16':V16, 'V17': V17,'V18': V18, 'V19': V19,'V20': V20,
'V21': V21, 'V22': V22,'V23': V23, 'V24': V24,'V25': V25,'V26': V26, 'V27': V27, 'V28': V28, 'Amount': Amount}


df=pd.DataFrame.from_dict([my_dict])


Button = st.button("Predict")

if Button:
    filename = 'scaler_fraud.pkl'
    model_fraud = load_model('fraud.h5')
    scaler_fraud = pickle.load(open("scaler_fraud", "rb"))
    df[["Time","Amount" ]] = scaler_fraud.transform(df[["Time", "Amount"]])
    pred = (model_fraud.predict(df) < 0.5).astype("int32")

    st.success('Fraud : {}'.format(pred[0]))


    

