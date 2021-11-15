import streamlit as st
from predict_page import show_predict_page
from statement_predict import show_statement_predict
from show_page import show_show_page
page = st.sidebar.selectbox("Explore Or Predict Or Show", ("statement predict","tweets Predict", "Show"))

if page == "statement predict":
    show_statement_predict()
elif page == "tweets Predict":
    show_predict_page()
elif page == "Show" : 
    show_show_page()
