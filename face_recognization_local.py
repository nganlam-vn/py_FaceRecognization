import tkinter as tk
from tkinter import Label, Button
import cv2
import numpy as np
from tensorflow.keras import models
from PIL import Image, ImageTk

# Load the face recognition model and face detector
class_name = [ 'HuuNghia', 'CongToan','KimNgan', 'HoangPhat', 'NganLam','DuyThien',]
saved_model = models.load_model('./Recognition_model.h5')
face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        
        # Create and place start and stop buttons
        self.start_button = Button(root, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT)
        
        self.stop_button = Button(root, text="Stop Camera", command=self.stop_camera)
        self.stop_button.pack(side=tk.RIGHT)
        
        # Create a label to display the video feed
        self.video_label = Label(root)
        self.video_label.pack()

        # Initialize video capture
        self.camera = cv2.VideoCapture(0)
        self.running = False

    def start_camera(self):
        self.running = True
        self.update_frame()

    def stop_camera(self):
        self.running = False

    def update_frame(self):
        if self.running:
            _, frame = self.camera.read()
            
            frame = cv2.flip(frame, 1) # flip video image vertically
            faces = face_detector.detectMultiScale(frame, 1.3, 5)
            for x, y, w, h in faces:
                roi = cv2.resize(frame[y:y+h, x:x+w], (64, 64))
                result = np.argmax(saved_model.predict(roi.reshape(-1, 64, 64, 3)))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, class_name[result], (x+15, y-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 2)

            # Convert frame to RGB and then to ImageTk format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.root.after(10, self.update_frame)
        else:
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
