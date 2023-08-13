import streamlit as st

st.header('Demo 📹')
st.text(' ')
video_file = open('app.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)