import streamlit as st
import pickle
import joblib
import pandas as pd

model = joblib.load('./best_nb_model.pkl')

@st.cache()
def prediction_threshold(text):

    pred_subr        = model.predict(pd.Series(text))
    pred_thresh_subr = model.predict_proba(pd.Series(text))

    if (pred_thresh_subr[0][0] or pred_thresh_subr[0][1]) > 0.75:
        if pred_subr[0] == 1:
            subr = "depressed"
        else:
            subr = "anxiety"
    else:
        subr = "indeterminite"

    return subr

st.title("How are you feeling?")

page = st.sidebar.selectbox("Select a page", ("About", "Make a prediction"))

if page == "About":
        st.write("Can we guess how you're feeling?")

if page == "Make a prediction":
    st.write("Can we tell if you're anxious, depressed, or indeterminite:")
    user_response = st.text_input("Where are you going?", value="I am sad")

    if st.button("My mood?"):
        mood = prediction_threshold(user_response)
        #lambda x: 1 if user_response.split() in ["suicide", "suicidal", "kill"]
        #if flag == 1:
            #st.write("[Help is available\
            #           Speak with someone today]\
            #           (https://suicidepreventionlifeline.org/)")
        st.write(f"You appear to be {mood}.")
