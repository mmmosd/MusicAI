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
# st.selectbox("sfds")
# st.subheader("")

music = model.Generate_Music('music', volume=25)

st.audio(music, format="audio/wav", start_time=0, sample_rate=44100)