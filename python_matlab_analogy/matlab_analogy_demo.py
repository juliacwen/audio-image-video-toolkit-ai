#!/usr/bin/env python3
"""
------------------------------------------------------------------------------
Python and MATLAB Analogy Demo Launcher
------------------------------------------------------------------------------
Author: Julia Wen (wendigilane@gmail.com)
Date: 11-10-2025
Description:
    Main Streamlit app to launch audio, image, and other demos.
    Calls src.audio_processing_demo.run() for audio section.
------------------------------------------------------------------------------
"""

import streamlit as st
from src import audio_processing_demo

st.set_page_config(page_title="Python MATLAB Analogy", layout="wide")

# ------------------- CSS for styling -------------------
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
div[role="radiogroup"] > label {
    background-color: #2196F3;
    color: #004080;
    font-weight: bold;
    padding: 8px 16px;
    margin-right: 5px;
    border-radius: 6px;
    font-size: 18px;
    cursor: pointer;
}
div[role="radiogroup"] > label[data-baseweb="true"]:has(input:checked) {
    background-color: #0073e6;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Main title -------------------
st.markdown(
    '<h1 style="font-size:36px; line-height:1.2; margin-bottom:0.5rem;">Python MATLAB Analogy Demo</h1>',
    unsafe_allow_html=True
)

# ------------------- Tabs -------------------
tabs = ["Audio Processing", "Image Processing", "Computer Vision", "Data Analysis"]
selected_tab = st.radio(
    label="Demo Selection",
    options=tabs,
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

# ------------------- Tab content -------------------
if selected_tab == "Audio Processing":
    st.markdown(
        '<h2 style="font-size:20px; line-height:1.2; margin-top:0.5rem; margin-bottom:0.5rem;">Audio Processing (MATLAB-style)</h2>',
        unsafe_allow_html=True
    )
    # Calls the full audio processing demo with synthetic/upload handling
    audio_processing_demo.run()
else:
    st.markdown(
        f'<h2 style="font-size:20px; line-height:1.2; margin-top:0.5rem; margin-bottom:0.5rem;">{selected_tab} demo coming soon.</h2>',
        unsafe_allow_html=True
    )

