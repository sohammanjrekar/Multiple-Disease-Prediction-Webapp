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
import streamlit as st
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt


# loading the models
diabetes_model = pickle.load(open("../models/diabetes_model.sav", "rb"))
heart_model = pickle.load(open("../models/heart_disease_model.sav", "rb"))
parkinson_model = pickle.load(open("../models/parkinsons_model.sav", "rb"))
liver_model = pickle.load(open("../models/liver.sav", "rb"))
jaundice_model=pickle.load(open("../models/jaundice.sav", "rb"))
hepatitis_model=pickle.load(open("../models/hepatitis.sav", "rb"))



# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Disease Prediction',
        'Diabetes prediction',
        'Heart disease prediction',
        'Parkison prediction',
        'Liver prediction',
        'Jaundice prediction',
        'Hepatitis prediction',
        'Dashboard',
        'Blogs'
    ],
        icons=['','activity', 'heart', 'person','person','person','person','bar-chart-fill'],
        default_index=0)



# Diabetes prediction page
if selected == 'Diabetes prediction':  # pagetitle
    st.title("Diabetes disease prediction")
    image = Image.open('d3.jpg')
    st.image(image, caption='diabetes disease prediction')
    # columns
    # no inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnencies")
    with col2:
        Glucose = st.number_input("Glucose level")
    with col3:
        BloodPressure = st.number_input("Blood pressure  value")
    with col1:

        SkinThickness = st.number_input("Sckinthickness value")

    with col2:

        Insulin = st.number_input("Insulin value ")
    with col3:
        BMI = st.number_input("BMI value")
    with col1:
        DiabetesPedigreefunction = st.number_input(
            "Diabetespedigreefunction value")
    with col2:

        Age = st.number_input("AGE")

    # code for prediction
    diabetes_dig = ''

    # button
    if st.button("Diabetes test result"):
        diabetes_prediction=[[]]
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
        if diabetes_prediction[0] == 1:
            diabetes_dig = "we are really sorry to say but it seems like you are Diabetic."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            diabetes_dig = 'Congratulation,You are not diabetic'
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name+' , ' + diabetes_dig)
        
        
# Heart prediction page
if selected == 'Heart disease prediction':
    st.title("Heart disease prediction")
    image = Image.open('heart2.jpg')
    st.image(image, caption='heart failuire')
    # age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
    # columns
    # no inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
    with col2:
        sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            sex = 1
        elif value == "female":
            sex = 0
    with col3:
        cp=0
        display = ("typical angina","atypical angina","non — anginal pain","asymptotic")
        options = list(range(len(display)))
        value = st.selectbox("Chest_Pain Type", options, format_func=lambda x: display[x])
        if value == "typical angina":
            cp = 0
        elif value == "atypical angina":
            cp = 1
        elif value == "non — anginal pain":
            cp = 2
        elif value == "asymptotic":
            cp = 3
    with col1:
        trestbps = st.number_input("Resting Blood Pressure")

    with col2:

        chol = st.number_input("Serum Cholestrol")
    
    with col3:
        restecg=0
        display = ("normal","having ST-T wave abnormality","left ventricular hyperthrophy")
        options = list(range(len(display)))
        value = st.selectbox("Resting ECG", options, format_func=lambda x: display[x])
        if value == "normal":
            restecg = 0
        elif value == "having ST-T wave abnormality":
            restecg = 1
        elif value == "left ventricular hyperthrophy":
            restecg = 2

    with col1:
        exang=0
        thalach = st.number_input("Max Heart Rate Achieved")
   
    with col2:
        oldpeak = st.number_input("ST depression induced by exercise relative to rest")
    with col3:
        slope=0
        display = ("upsloping","flat","downsloping")
        options = list(range(len(display)))
        value = st.selectbox("Peak exercise ST segment", options, format_func=lambda x: display[x])
        if value == "upsloping":
            slope = 0
        elif value == "flat":
            slope = 1
        elif value == "downsloping":
            slope = 2
    with col1:
        ca = st.number_input("Number of major vessels (0–3) colored by flourosopy")
    with col2:
        thal=0
        display = ("normal","fixed defect","reversible defect")
        options = list(range(len(display)))
        value = st.selectbox("thalassemia", options, format_func=lambda x: display[x])
        if value == "normal":
            thal = 0
        elif value == "fixed defect":
            thal = 1
        elif value == "reversible defect":
            thal = 2
    with col3:
        agree = st.checkbox('Exercise induced angina')
        if agree:
            exang = 1
        else:
            exang=0
    with col1:
        agree1 = st.checkbox('fasting blood sugar > 120mg/dl')
        if agree1:
            fbs = 1
        else:
            fbs=0
    # code for prediction
    heart_dig = ''
    

    # button
    if st.button("Heart test result"):
        heart_prediction=[[]]
        # change the parameters according to the model
        
        # b=np.array(a, dtype=float)
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_dig = 'we are really sorry to say but it seems like you have Heart Disease.'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            
        else:
            heart_dig = "Congratulation , You don't have Heart Disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name +' , ' + heart_dig)







if selected == 'Parkison prediction':
    st.title("Parkison prediction")
    image = Image.open('p1.jpg')
    st.image(image, caption='parkinsons disease')
  # parameters
#    name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	MDVP:Shimmer(dB)	Shimmer:APQ3	Shimmer:APQ5	MDVP:APQ	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
   # change the variables according to the dataset used in the model
    name = st.text_input("Name:")
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
    if st.button("Parkinson test result"):
        parkinson_prediction=[[]]
        # change the parameters according to the model
        parkinson_prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,  RPDE, DFA, spread1, spread2, D2, PPE]])

        if parkinson_prediction[0] == 1:
            parkinson_dig = 'we are really sorry to say but it seems like you have Parkinson disease'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            parkinson_dig = "Congratulation , You don't have Parkinson disease"
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name+' , ' + parkinson_dig)


# Liver prediction page
if selected == 'Liver prediction':  # pagetitle
    st.title("Liver disease prediction")
    image = Image.open('liver.jpg')
    st.image(image, caption='Liver disease prediction.')
    # columns
    # no inputs from the user
# st.write(info.astype(int).info())
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col2:
        age = st.number_input("Entre your age") # 2 
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Entre your Aspartate_Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col3:
        Albumin = st.number_input("Entre your Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Entre your Albumin_and_Globulin_Ratio") # 10 
    # code for prediction
    liver_dig = ''

    # button
    if st.button("Liver test result"):
        liver_prediction=[[]]
        liver_prediction = liver_model.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
        if liver_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            liver_dig = "we are really sorry to say but it seems like you have liver disease."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            liver_dig = "Congratulation , You don't have liver disease."
        st.success(name+' , ' + liver_dig)




# hepatitis prediction page
if selected == 'Hepatitis prediction':  # pagetitle
    st.title("Hepatitis disease prediction")
    image = Image.open('h.png')
    st.image(image, caption='Hepatitis disease prediction')
    # columns
    # no inputs from the user
# st.write(info.astype(int).info())
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Entre your age   ") # 2 
    with col2:
        sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            sex = 0
        elif value == "female":
            sex = 1
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Entre your Aspartate_Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col3:
        Albumin = st.number_input("Entre your Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Entre your Albumin_and_Globulin_Ratio") # 10 
    # code for prediction
    hepatitis_dig = ''

    # button
    if st.button("Hepatitis test result"):
        hepatitis_prediction=[[]]
        hepatitis_prediction = hepatitis_model.predict([[age,sex,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
        if hepatitis_prediction[0] == 1:
            hepatitis_dig = "we are really sorry to say but it seems like you are Hepatitic."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            hepatitis_dig = 'Congratulation,You are not Hepatitic.'
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name+' , ' + hepatitis_dig)








# jaundice prediction page
if selected == 'Jaundice prediction':  # pagetitle
    st.title("Jaundice disease prediction")
    image = Image.open('j.jpg')
    st.image(image, caption='Jaundice disease prediction')
    # columns
    # no inputs from the user
# st.write(info.astype(int).info())
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Entre your age   ") # 2 
    with col2:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col2:
        Albumin = st.number_input("Entre your Albumin") # 9 
    # code for prediction
    jaundice_dig = ''

    # button
    if st.button("Jaundice test result"):
        jaundice_prediction=[[]]
        jaundice_prediction = jaundice_model.predict([[age,Sex,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Total_Protiens,Albumin]])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
        if jaundice_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            jaundice_dig = "we are really sorry to say but it seems like you have Jaundice."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            jaundice_dig = "Congratulation , You don't have Jaundice."
        st.success(name+' , ' + jaundice_dig)




# multiple disease prediction
if selected == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')





if selected == 'Dashboard':  # pagetitle
    st.title("Dashboard")
    #loading csv files
    diabetes_data = pd.read_csv("../Datasets/diabetes.csv")
    heart_data = pd.read_csv("../Datasets/heart.csv")
    parkinsons_data = pd.read_csv("../Datasets/parkinsons.csv")
    liver_data = pd.read_csv("../Datasets/Liver_dataset.csv")
    hepatities_data = pd.read_csv("../Datasets/hepatitis.csv")
    jaundice_data = pd.read_csv("../Datasets/jaundice_dataset.csv")
    
    
    
    
    
    select = st.selectbox('Select disease', ['Diabetes', 'Heart', 'Parkinsons','Liver','Jaundice','Hepatitis'],key='2')
    st.title("Data visualization")
    
    
    
    if select=='Diabetes':
        select = st.selectbox('Select Data visulization ', ['Line Plot', 'Violin Plot', 'Strip Plot','count plot','heatmap plot'],key='3')
        for i in ['Pregnancies'	,'Glucose',	'BloodPressure',	'SkinThickness','Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age']:
            if select=='Line Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.lineplot(x = diabetes_data[i], y = diabetes_data['Outcome'])
                st.pyplot(fig)
            if select=='Violin Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.violinplot(x = diabetes_data[i], y = diabetes_data['Outcome'])
                st.pyplot(fig) 
            if select=='Strip Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.stripplot(x = diabetes_data[i], y = diabetes_data['Outcome'])
                st.pyplot(fig) 
            if select=='count plot':
                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x = diabetes_data[i])
                st.pyplot(fig)
        if select=='heatmap plot':
            fig, ax = plt.subplots()
            sns.heatmap(diabetes_data.corr(), ax=ax)
            st.write(fig)
            
            
            
    if select=='Heart':
        select = st.selectbox('Select Data visulization ', ['Line Plot', 'Violin Plot', 'Strip Plot','count plot','heatmap plot'],key='3')
        for i in ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']:
            if select=='Line Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.lineplot(x =  heart_data[i], y =  heart_data['target'])
                st.pyplot(fig)
            if select=='Violin Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.violinplot(x =  heart_data[i], y =  heart_data['target'])
                st.pyplot(fig) 
            if select=='Strip Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.stripplot(x =  heart_data[i], y =  heart_data['target'])
                st.pyplot(fig) 
            if select=='count plot':
                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x =  heart_data[i])
                st.pyplot(fig)
        if select=='heatmap plot':
            fig, ax = plt.subplots()
            sns.heatmap(heart_data.corr(), ax=ax)
            st.write(fig)      
        
    if select=='Parkinsons':
        select = st.selectbox('Select Data visulization ',['Line Plot', 'Violin Plot', 'Strip Plot','count plot','heatmap plot'],key='3') 
        for i in ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']:
            if select=='Line Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.lineplot(x =  parkinsons_data[i], y =  parkinsons_data['status'])
                st.pyplot(fig)
            if select=='Violin Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.violinplot(x =  parkinsons_data[i], y =  parkinsons_data['status'])
                st.pyplot(fig) 
            if select=='Strip Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.stripplot(x =  parkinsons_data[i], y =  parkinsons_data['status'])
                st.pyplot(fig) 
            if select=='count plot':
                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x = parkinsons_data[i])
                st.pyplot(fig)
        if select=='heatmap plot':
            fig, ax = plt.subplots()
            sns.heatmap(parkinsons_data.corr(), ax=ax)
            st.write(fig)                     
                        
                        
                        
    

    if select=='Liver':
        select = st.selectbox('Select Data visulization ', ['Line Plot', 'Violin Plot', 'Strip Plot','count plot','heatmap plot'],key='3')
        for i in ['Age','Sex','Total_Bilirubin',
 'Direct_Bilirubin',
 'Alkaline_Phosphotase',
 'Alamine_Aminotransferase',
 'Aspartate_Aminotransferase',
 'Total_Protiens',
 'Albumin',
 'Albumin_and_Globulin_Ratio']:
            if select=='Line Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.lineplot(x = liver_data[i], y = liver_data['Result'])
                st.pyplot(fig)
            if select=='Violin Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.violinplot(x = liver_data[i], y = liver_data['Result'])
                st.pyplot(fig) 
            if select=='Strip Plot':
                fig = plt.figure(figsize=(10, 4))
                sns.stripplot(x = liver_data[i], y = liver_data['Result'])
            if select=='count plot':
                fig = plt.figure(figsize=(10, 4))
                sns.countplot(x = liver_data[i])
                st.pyplot(fig)
        if select=='heatmap plot':
            fig, ax = plt.subplots()
            sns.heatmap(liver_data.corr(), ax=ax)
            st.write(fig)

    if select=='Jaundice':
            select = st.selectbox('Select Data visulization ', ['Line Plot', 'Violin Plot', 'Strip Plot','count plot','heatmap plot'],key='3')
            for i in ['AGE',
 'GENDER',
 'Total_Bilirubin',
 'Direct_Bilirubin',
 'Alkaline_Phosphotase',
 'Alamine_Aminotransferase',
 'Total_Protiens',
 'Albumin',
 ]:
                if select=='Line Plot':
                    fig = plt.figure(figsize=(10, 4))
                    sns.lineplot(x =  jaundice_data[i], y =  jaundice_data['jaundice'])
                    st.pyplot(fig)
                if select=='Violin Plot':
                    fig = plt.figure(figsize=(10, 4))
                    sns.violinplot(x =  jaundice_data[i], y =  jaundice_data['jaundice'])
                    st.pyplot(fig) 
                if select=='Strip Plot':
                    fig = plt.figure(figsize=(10, 4))
                    sns.stripplot(x =  jaundice_data[i], y =  jaundice_data['jaundice'])
                    st.pyplot(fig) 
                if select=='count plot':
                    fig = plt.figure(figsize=(10, 4))
                    sns.countplot(x =  jaundice_data[i])
                    st.pyplot(fig)
            if select=='heatmap plot':
                fig, ax = plt.subplots()
                sns.heatmap(jaundice_data.corr(), ax=ax)
                st.write(fig)





    if select=='Hepatitis':
            select = st.selectbox('Select Data visulization ', ['Line Plot', 'Violin Plot', 'Strip Plot','count plot','heatmap plot'],key='3')
            for i in ['AGE',
    'GENDER',
    'TOTAL_BILIRUBIN',
    'DIRECT_BILIRUBIN',
    'ALKALINE_PHOSPHOTASE',
    'ALAMINE_AMINOTRANSFERASE',
    'ASPARTATE_AMINOTRANSFERASE',
    'TOTAL_PROTEINS',
    'ALBUMIN',
    'ALBUMIN_AND_GLOBULIN_RATIO',
    ]:
                if select=='Line Plot':
                    fig = plt.figure(figsize=(10, 4))
                    sns.lineplot(x = hepatities_data[i], y = hepatities_data['DATASET'])
                    st.pyplot(fig)
                if select=='Violin Plot':
                    fig = plt.figure(figsize=(10, 4))
                    sns.violinplot(x = hepatities_data[i], y = hepatities_data['DATASET'])
                    st.pyplot(fig) 
                if select=='Strip Plot':
                    fig = plt.figure(figsize=(10, 4))
                    sns.stripplot(x =hepatities_data[i], y = hepatities_data['DATASET'])
                    st.pyplot(fig) 
                if select=='count plot':
                    fig = plt.figure(figsize=(10, 4))
                    sns.countplot(x = hepatities_data[i])
                    st.pyplot(fig)
            if select=='heatmap plot':
                fig, ax = plt.subplots()
                sns.heatmap(hepatities_data.corr(), ax=ax)
                st.write(fig)




















import sqlite3
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nlp

conn = sqlite3.connect("data.db")
c = conn.cursor()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS blogtable(author TEXT,title TEXT,article TEXT,postdate DATE)')

def add_data(author,title,article,postdate):
    c.execute('INSERT INTO blogtable(author,title,article,postdate) VALUES (?,?,?,?)',(author,title,article,postdate))
    conn.commit()

def view_all_notes():
	c.execute('SELECT * FROM blogtable')
	data = c.fetchall()
	return data

def view_all_titles():
	c.execute('SELECT DISTINCT title FROM blogtable')
	data = c.fetchall()
	return data

def get_blog_by_title(title):
	c.execute('SELECT * FROM blogtable WHERE title="{}"'.format(title))
	data = c.fetchall()
	return data

def get_blog_by_author(author):
	c.execute('SELECT * FROM blogtable WHERE author="{}"'.format(author))
	data = c.fetchall()
	return data

def delete_data(title):
	c.execute('DELETE FROM blogtable WHERE title="{}"'.format(title))
	conn.commit()

# Reading Time
def readingTime(mytext):
	total_words = len([ token for token in mytext.split(" ")])
	estimatedTime = total_words/200.0
	return estimatedTime

def analyze_text(text):
	return nlp(text)

# Layout templete
title_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<h6>Author:{}</h6>
	<br/>
	<br/>	
	<p style="text-align:justify">{}</p>
	</div>
	"""
article_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<h6>Author:{}</h6> 
	<h6>Post Date: {}</h6>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>
	<p style="text-align:justify">{}</p>
	</div>
	"""
head_message_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
	<h4 style="color:white;text-align:center;">{}</h1>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;">
	<h6>Author:{}</h6> 		
	<h6>Post Date: {}</h6>		
	</div>
	"""
full_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<p style="text-align:justify;color:white;padding:10px">{}</p>
	</div>
	"""

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

def main():
    if selected == 'Blogs':
        st.title("Disease Blogs")

        menu = ["Home","View Posts","Add Post","Search","Manage Blog"]
        choice = st.sidebar.selectbox("Menu",menu)

        if choice == "Home":
            st.subheader("Home")
            result = view_all_notes()

            for i in result:
                b_author = i[0]
                b_title = i[1]
                b_article = str(i[2])[0:100]
                b_post_date = i[3]
                st.markdown(title_temp.format(b_title,b_author,b_article,b_post_date),unsafe_allow_html=True)

        elif choice == "View Posts":
            st.subheader("View Posts")
            all_titles = [i[0] for i in view_all_titles()]
            postlist = st.sidebar.selectbox("View Posts",all_titles)
            post_result = get_blog_by_title(postlist)
            for i in post_result:
                b_author = i[0]
                b_title = i[1]
                b_article = i[2]
                b_post_date = i[3]
                st.text("Reading Time:{} minutes".format(readingTime(str(i[2]))))
                st.markdown(head_message_temp.format(i[1], i[0], i[3]), unsafe_allow_html=True)
                st.markdown(full_message_temp.format(i[2]), unsafe_allow_html=True)

        elif choice == "Add Post":
            st.subheader("Add Post")
            create_table()

            blog_author = st.text_input("Enter the name of author",max_chars=50)
            blog_title = st.text_input("Enter Post title")
            blog_article = st.text_area("Post article here", height=200)
            blog_post_date = st.date_input("Date")

            if st.button("Add"):
                add_data(blog_author,blog_title,blog_article,blog_post_date)
                st.success("Post : {} Saved".format(blog_title))


        elif choice == "Search":

            st.subheader("Search Articles")

            search_term = st.text_input("Enter Term")

            search_choice = st.radio("Field to Search", ("title", "author"))

            if st.button('Search'):

                if search_choice == "title":

                    article_result = get_blog_by_title(search_term)

                elif search_choice == "author":

                    article_result = get_blog_by_author(search_term)

                # Preview Articles

                for i in article_result:
                    st.text("Reading Time:{} minutes".format(readingTime(str(i[2]))))

                    # st.write(article_temp.format(i[1],i[0],i[3],i[2]),unsafe_allow_html=True)

                    st.write(head_message_temp.format(i[1], i[0], i[3]), unsafe_allow_html=True)

                    st.write(full_message_temp.format(i[2]), unsafe_allow_html=True)



        elif choice == "Manage Blog":

            st.subheader("Manage Blog")

            result = view_all_notes()
            clean_db = pd.DataFrame(result,columns=["Author","Title","Article"," Post Date"])
            st.dataframe(clean_db)
            unique_list = [i[0] for i in view_all_titles()]
            delete_by_title = st.selectbox("Select Title", unique_list)
            if st.button("Delete"):
                delete_data(delete_by_title)
                st.warning("Deleted: '{}'".format(delete_by_title))



if __name__ == '__main__':
    main()
