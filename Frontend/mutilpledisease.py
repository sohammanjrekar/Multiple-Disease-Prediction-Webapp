import streamlit as st


from streamlit_option_menu import option_menu
import pickle


# loading the models
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
heart_model = pickle.load(open("heart_disease_model.sav", "rb"))
parkinson_model = pickle.load(open("parkinsons_model.sav", "rb"))

# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'diabetes prediction',
        'heart disease prediction',
        'parkison prediction'
    ],
        icons=['activity', 'heart', 'person'],
        default_index=0)

# Diabetes prediction page
if selected == 'diabetes prediction':  # pagetitle
    st.title("Diabetes disease prediction")

    # columns
    # no inputs from the user

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnencies")
    with col2:
        Glucose = st.text_input("Glucose level")
    with col3:
        BloodPressure = st.text_input("Blood pressure  value")
    with col1:

        SkinThickness = st.text_input("Sckinthickness value")

    with col2:

        Insulin = st.text_input("Insulin value ")
    with col3:
        BMI = st.text_input("BMI value")
    with col1:
        DiabetesPedigreefunction = st.text_input(
            "Diabetespedigreefunction value")
    with col2:

        Age = st.text_input("AGE")

    # code for prediction
    diabetes_dig = ''

    # button
    if st.button("Diabetes test result"):
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
    if diabetes_prediction[0] == 1:
        diabetes_dig = 'The person is Diabetic'
    else:
        diabetes_dig = 'THe person is not Diabetic'
    st.success(diabetes_dig)

if selected == 'heart disease prediction':
    st.title("Heart disease prediction")

    # age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
    # columns
    # no inputs from the user

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("AGE")
    with col2:
        sex = st.text_input("sex")
    with col3:
        cp = st.text_input("cp value")
    with col1:
        trestbps = st.text_input("trestbps value")

    with col2:

        chol = st.text_input("chol value ")
    with col3:
        fbs = st.text_input("fbs value")
    with col1:
        restecg = st.text_input("restecg value")
    with col2:
        thalach = st.text_input("thalach value")
    with col3:
        exang = st.text_input("exang value")
    with col1:
        oldpeak = st.text_input("oldpeak value")
    with col2:
        slope = st.text_input("slope value")
    with col3:
        ca = st.text_input("ca value")
    with col1:
        thal = st.text_input("Thal value")

    # code for prediction
    heart_dig = ''

    # button
    if st.button("Heart test result"):
        # change the parameters according to the model
        heart_prediction = heart_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_dig = 'The person have heart disease'
        else:
            heart_dig = 'THe person does not have heart disease'
    st.success(heart_dig)


if selected == 'parkison prediction':
    st.title("Parkison prediction")
  # parameters
#    name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	MDVP:Shimmer(dB)	Shimmer:APQ3	Shimmer:APQ5	MDVP:APQ	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
   # change the variables according to the dataset used in the model

    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP = st.text_input("MDVP:Fo(Hz)")
    with col2:
        MDVPFIZ = st.text_input("MDVP:Fhi(Hz)")
    with col3:
        MDVPFLO = st.text_input("MDVP:Flo(Hz)")
    with col1:
        MDVPJITTER = st.text_input("MDVP:Jitter(%)")
    with col2:
        MDVPJitterAbs = st.text_input("MDVP:Jitter(Abs)")
    with col3:
        MDVPRAP = st.text_input("MDVP:RAP")

    with col2:

        MDVPPPQ = st.text_input("MDVP:PPQ ")
    with col3:
        JitterDDP = st.text_input("Jitter:DDP")
    with col1:
        MDVPShimmer = st.text_input("MDVP:Shimmer")
    with col2:
        MDVPShimmer_dB = st.text_input("MDVP:Shimmer(dB)")
    with col3:
        Shimmer_APQ3 = st.text_input("Shimmer:APQ3")
    with col1:
        ShimmerAPQ5 = st.text_input("Shimmer:APQ5")
    with col2:
        MDVP_APQ = st.text_input("MDVP:APQ")
    with col3:
        ShimmerDDA = st.text_input("Shimmer:DDA")
    with col1:
        NHR = st.text_input("NHR")
    with col2:
        HNR = st.text_input("HNR")
  
    with col2:
        RPDE = st.text_input("RPDE")
    with col3:
        DFA = st.text_input("DFA")
    with col1:
        spread1 = st.text_input("spread1")
    with col1:
        spread2 = st.text_input("spread2")
    with col3:
        D2 = st.text_input("D2")
    with col1:
        PPE = st.text_input("PPE")

    # code for prediction
    parkinson_dig = ''

    # button
    if st.button("Parkinson test result"):
        # change the parameters according to the model
     parkinson_prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,
                                                   MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,  RPDE, DFA, spread1, spread2, D2, PPE]])

    if parkinson_prediction[0] == 1:
        parkinson_dig = 'The person have Parkinson disease'
    else:
        parkinson_dig = 'THe person does not have Parkinson disease'
    st.success(parkinson_dig)

