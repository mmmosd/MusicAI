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
interpolation_count = st.slider('Interpolation Count 🎶', 1, 20)

if button:
    st.write('Generating...')
    music = model.Generate(saved_model_name='Generator_epoch_100.pt', interpolation_count=interpolation_count, volume=25)
    st.audio(music, format="audio/wav", start_time=0, sample_rate=44100)