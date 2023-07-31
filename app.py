import streamlit as st
import base64
import model
import converter
import os

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


# autoplay_audio(filepath)
# st.audio(Result_Audio, format="audio/wav", start_time=0, sample_rate=44100)