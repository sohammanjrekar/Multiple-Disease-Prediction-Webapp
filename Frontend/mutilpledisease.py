import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff


# loading the models
diabetes_model = pickle.load(open("../models/diabetes_model.sav", "rb"))
heart_model = pickle.load(open("../models/heart_disease_model.sav", "rb"))
parkinson_model = pickle.load(open("../models/parkinsons_model.sav", "rb"))


# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Diabetes Prediction',
        'Heart Disease Prediction',
        'Parkison Prediction',
        'Dashboard'
    ],
        icons=['activity', 'heart', 'person','list-task'],
        default_index=0)

# Diabetes prediction page
if selected == 'Diabetes Prediction':  # pagetitle
    st.title("Diabetes Disease Prediction")
    image = Image.open('d3.jpg')
    st.image(image, caption='DIABETES DISEASE PREDICTION')
    # columns
    # no inputs from the user

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnancies")
    with col2:
        Glucose = st.number_input("Glucose Level")
    with col3:
        BloodPressure = st.number_input("Blood Pressure Value")
    with col1:

        SkinThickness = st.number_input("Skin Thickness Value")

    with col2:

        Insulin = st.number_input("Insulin Value ")
    with col3:
        BMI = st.number_input("BMI Value")
    with col1:
        DiabetesPedigreefunction = st.number_input(
            "Diabetes Pedigree Function Value")
    with col2:

        Age = st.number_input("AGE")

    # code for prediction
    diabetes_dig = ''

    # button
    if st.button("DIABETES TEST RESULT"):
        diabetes_prediction=[[]]
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])

        # after the prediction is done if the Value in the list at index is 0 is 1 then the person is diabetic
        if diabetes_prediction[0] == 1:
            diabetes_dig = 'THE PERSON IS DIABETIC.'
        else:
            diabetes_dig = 'THE PERSON IS NOT DIABETIC.'
        st.success(diabetes_dig)

if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction")
    image = Image.open('heart2.jpg')
    st.image(image, caption='HEART FAILURE')
    # age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
    # columns
    # no inputs from the user

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("AGE")
    with col2:
        sex = st.number_input("SEX")
    with col3:
        cp = st.number_input("CP Value")
    with col1:
        trestbps = st.number_input("TRESTBPS Value")
    with col2:
        chol = st.number_input("CHOL Value ")
    with col3:
        fbs = st.number_input("FBS Value")
    with col1:
        restecg = st.number_input("RESTECG Value")
    with col2:
        thalach = st.number_input("THALACH Value")
    with col3:
        exang = st.number_input("EXANG Value")
    with col1:
        oldpeak = st.number_input("OLDPEAK Value")
    with col2:
        slope = st.number_input("SLOPE Value")
    with col3:
        ca = st.number_input("CA Value")
    with col1:
        thal = st.number_input("THAL Value")

    # code for prediction
    heart_dig = ''

    # button
    if st.button("HEART TEST RESULT"):
        heart_prediction=[[]]
        # change the parameters according to the model
        
        # b=np.array(a, dtype=float)
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_dig = 'THE PERSON HAVE HEART DISEASE.'
        else:
            heart_dig = 'THE PERSON DOES NOT HAVE HEART DISEASE.'
        st.success(heart_dig)


if selected == 'Parkison Prediction':
    st.title("Parkison Prediction")
    image = Image.open('p1.jpg')
    st.image(image, caption='PARKINSONS DISEASE')
  # parameters
#    name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	MDVP:Shimmer(dB)	Shimmer:APQ3	Shimmer:APQ5	MDVP:APQ	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
   # change the variables according to the dataset used in the model

    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP = st.number_input("MDVP:Fo(Hz)")
    with col2:
        MDVPFIZ = st.number_input("MDVP:Fhi(Hz)")
    with col3:
        MDVPFLO = st.number_input("MDVP:Flo(Hz)")
    with col1:
        MDVPJITTER = st.number_input("MDVP:Jitter(%)")
    with col2:
        MDVPJitterAbs = st.number_input("MDVP:Jitter(Abs)")
    with col3:
        MDVPRAP = st.number_input("MDVP:RAP")
    with col2:
        MDVPPPQ = st.number_input("MDVP:PPQ ")
    with col3:
        JitterDDP = st.number_input("Jitter:DDP")
    with col1:
        MDVPShimmer = st.number_input("MDVP:Shimmer")
    with col2:
        MDVPShimmer_dB = st.number_input("MDVP:Shimmer(dB)")
    with col3:
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3")
    with col1:
        ShimmerAPQ5 = st.number_input("Shimmer:APQ5")
    with col2:
        MDVP_APQ = st.number_input("MDVP:APQ")
    with col3:
        ShimmerDDA = st.number_input("Shimmer:DDA")
    with col1:
        NHR = st.number_input("NHR")
    with col2:
        HNR = st.number_input("HNR")
    with col2:
        RPDE = st.number_input("RPDE")
    with col3:
        DFA = st.number_input("DFA")
    with col1:
        spread1 = st.number_input("spread1")
    with col1:
        spread2 = st.number_input("spread2")
    with col3:
        D2 = st.number_input("D2")
    with col1:
        PPE = st.number_input("PPE")

    # code for prediction
    parkinson_dig = ''
    
    # button
    if st.button("PARKINSON TEST RESULT"):
        parkinson_prediction=[[]]
        # change the parameters according to the model
        parkinson_prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,  RPDE, DFA, spread1, spread2, D2, PPE]])

        if parkinson_prediction[0] == 1:
            parkinson_dig = 'THE PERSON HAVE PARKINSON DISEASE.'
        else:
            parkinson_dig = 'THE PERSON DOES NOT HAVE PARKINSON DISEASE.'
        st.success(parkinson_dig)
