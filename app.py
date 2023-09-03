import streamlit as st
import base64
import model
import converter
import os

st.set_page_config(
    page_icon="🎹",
    page_title="LOFIAI",
    layout="wide",
)

st.header("MUSIC_GENERATOR 🎷")

button = st.button('Generate')
interpolation_count = st.slider('Generate Count 🎶', 1, 50)

if button:
    st.write('Generating...')
    music = model.Generate_Music('music', interpolation_count=interpolation_count, volume=25)
    st.audio(music, format="audio/wav", start_time=0, sample_rate=44100)
    # st.image()