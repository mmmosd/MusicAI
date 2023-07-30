import streamlit as st
import model
import converter
import os

st.set_page_config(
    page_icon="🎹",
    page_title="LOFIAI",
    layout="wide",
)

st.header("LOFI-LOOP 🎷")
st.subheader("made by mmmosd")

spg = converter.Sound_To_Spectrogram(os.getcwd()+'\Sphere.mp3', 60)

fileName = 'SphereSPG.wav'
Audio = converter.Save_Spectrogram_Audio(spg, fileName, False)

st.audio(Audio, format="audio/wav", start_time=0, sample_rate=44100)