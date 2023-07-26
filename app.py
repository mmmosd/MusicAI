import subprocess
packages = str(subprocess.run('pip list', capture_output=True))
st.markdown(packages.replace('\\r\\n', '  \\\n'))

import streamlit as st
import model

model.printTFVersion()

st.set_page_config(
    page_icon="🎹",
    page_title="mmmosdMusicAI",
    layout="wide",
)

st.header("fdas")
st.subheader("fdsa")