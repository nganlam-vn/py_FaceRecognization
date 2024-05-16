import streamlit as st
import cv2
import numpy as np
from tensorflow.keras import models

st.set_page_config(
    page_title='AI SEMESTER REPORT', 
    layout='wide', 
    initial_sidebar_state='collapsed',
)

class_name = [ 'HuuNghia', 'CongToan','KimNgan', 'HoangPhat', 'NganLam','DuyThien',]
saved_model = models.load_model('.\Recognition_model.h5')
face_detector = cv2.CascadeClassifier('.\haarcascade_frontalface_alt.xml')

st.title("Face Recognition App")

run = st.button("Start Camera" )
    
stop = st.button("Stop Camera")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
while run:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1) # flip video image vertically
    # Rest of the code
    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        roi = cv2.resize(frame[y:y+h, x:x+w], (64, 64))
        result = np.argmax(saved_model.predict(roi.reshape(-1, 64, 64, 3)))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, class_name[result], (x+15, y-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 2)
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if stop:
    camera.release()
    cv2.destroyAllWindows()
