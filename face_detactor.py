import tkinter as tk
from tkinter import Label, Button
import cv2
import numpy as np
from tensorflow.keras import models
from PIL import Image, ImageTk

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0

while True:

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("./TRAIN/HoangPhat/HoangPhat_" + str(face_id) + '.' + str(count) + ".jpg", img[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 98:
        break
    elif count >= 100: # Take 30 face sample and stop video
         break
    
    print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()