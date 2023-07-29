import streamlit as st
import model
import converter
import os
import base64

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

st.set_page_config(
    page_icon="🎹",
    page_title="LOFIAI",
    layout="wide",
)

st.header("LOFI-LOOP 🎷")
st.subheader("made by mmmosd")

spg = converter.Sound_To_Spectrogram('Sphere.mp3', 60)

fileName = 'SphereSPG.wav'
Audio = converter.Save_Spectrogram_Audio(spg, fileName)

autoplay_audio(fileName)