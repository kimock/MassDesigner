import streamlit as st
import os
from PIL import Image

image = Image.open("mass_designer_logo.png")   # 같은 디렉토리

# 1. 시작
st.header("Mass Designer Operation")
st.image(image, width=300)

