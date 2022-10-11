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
        'diabetes prediction',
        'heart disease prediction',
        'parkison prediction',
        'Dashboard'
    ],
        icons=['activity', 'heart', 'person','list-task'],
        default_index=0)

# Diabetes prediction page
if selected == 'diabetes prediction':  # pagetitle
    st.title("Diabetes disease prediction")
    image = Image.open('d3.jpg')
    st.image(image, caption='diabetes disease prediction')
    # columns
    # no inputs from the user

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
            diabetes_dig = 'The person is Diabetic'
        else:
            diabetes_dig = 'THe person is not Diabetic'
        st.success(diabetes_dig)

if selected == 'heart disease prediction':
    st.title("Heart disease prediction")
    image = Image.open('heart2.jpg')
    st.image(image, caption='heart failuire')
    # age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target
    # columns
    # no inputs from the user

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("AGE")
    with col2:
        sex = st.number_input("sex")
    with col3:
        cp = st.number_input("cp value")
    with col1:
        trestbps = st.number_input("trestbps value")

    with col2:

        chol = st.number_input("chol value ")
    with col3:
        fbs = st.number_input("fbs value")
    with col1:
        restecg = st.number_input("restecg value")
    with col2:
        thalach = st.number_input("thalach value")
    with col3:
        exang = st.number_input("exang value")
    with col1:
        oldpeak = st.number_input("oldpeak value")
    with col2:
        slope = st.number_input("slope value")
    with col3:
        ca = st.number_input("ca value")
    with col1:
        thal = st.number_input("Thal value")

    # code for prediction
    heart_dig = ''
    

    # button
    if st.button("Heart test result"):
        heart_prediction=[[]]
        # change the parameters according to the model
        
        # b=np.array(a, dtype=float)
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_dig = 'The person have heart disease'
        else:
            heart_dig = 'THe person does not have heart disease'
        st.success(heart_dig)


if selected == 'parkison prediction':
    st.title("Parkison prediction")
    image = Image.open('p1.jpg')
    st.image(image, caption='parkinsons disease')
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
    if st.button("Parkinson test result"):
        parkinson_prediction=[[]]
        # change the parameters according to the model
        parkinson_prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,
                                                   MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,  RPDE, DFA, spread1, spread2, D2, PPE]])

    if parkinson_prediction[0] == 1:
        parkinson_dig = 'The person have Parkinson disease'
    else:
        parkinson_dig = 'THe person does not have Parkinson disease'
    st.success(parkinson_dig)















































# if selected == 'Dashboard':  # pagetitle
#     st.title("Dashboard")
#     #loading csv files
#     diabetes_data = pd.read_csv("../Datasets/diabetes.csv")
#     heart_data = pd.read_csv("../Datasets/heart.csv")
#     parkinsons_data = pd.read_csv("../Datasets/parkinsons.csv")
#     select = st.sidebar.selectbox('Select disease', ['Diabetes', 'heart', 'parkinsons'],key='2')
    
    
#     st.title("Information about disease")
#     if select == 'Diabetes':
#         for i in ['Pregnancies'	,'Glucose',	'BloodPressure',	'SkinThickness','Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age','Outcome']:
#             st.bar_chart([diabetes_data['Outcome'],diabetes_data[i]])
#             st.line_chart([diabetes_data['Outcome'],diabetes_data[i]])
#             st.area_chart([diabetes_data['Outcome'],diabetes_data[i]])

import sqlite3
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


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

        blog_author = st.number_input("Enter the name of author",max_chars=50)
        blog_title = st.number_input("Enter Post title")
        blog_article = st.text_area("Post article here", height=200)
        blog_post_date = st.date_input("Date")

        if st.button("Add"):
            add_data(blog_author,blog_title,blog_article,blog_post_date)
            st.success("Post : {} Saved".format(blog_title))


    elif choice == "Search":

        st.subheader("Search Articles")

        search_term = st.number_input("Enter Term")

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
                
            
            
           