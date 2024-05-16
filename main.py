import cv2
import numpy as np 
import os 
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras import layers


TRAIN_DIR = '.\TRAIN'
TEST_DIR = '.\TEST'

x_train = []
x_test = []

# dict = {'CongToan':[0,0,1], 'HuuNghia':[0,1,0], 
#         'KimNgan': [1,0,0],
#         'CongToan_test':[0,0,1], 'HuuNghia_test':[0,1,0], 
#         'KimNgan_test': [1,0,0],
#         }

dict = {'CongToan': [0,0,0,0,0,1], 'HuuNghia': [0,0,0,0,1,0], 'KimNgan': [0,0,0,1,0,0],
      'NganLam' : [0,0,1,0,0,0], 'HoangPhat' : [0,1,0,0,0,0], 'DuyThien' : [1,0,0,0,0,0],
      'CongToan_test': [0,0,0,0,0,1], 'HuuNghia_test': [0,0,0,0,1,0], 'KimNgan_test': [0,0,0,1,0,0],
      'NganLam_test' : [0,0,1,0,0,0], 'HoangPhat_test' : [0,1,0,0,0,0], 'DuyThien_test' : [1,0,0,0,0,0],}

def getData(data_dir, lst_data):
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            label = folder
            img = np.array(Image.open(file_path))
            lst_data.append((img, dict[label]))
    return lst_data

x_train = getData(TRAIN_DIR, x_train)
x_test = getData(TEST_DIR, x_test)

np.random.shuffle(x_train)
# train model

model_cnn = models.Sequential([
    layers.Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(256, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),

    layers.Flatten(input_shape = (64,64,3)),
    layers.Dense(3000, activation = 'relu'),
    layers.Dense(1000, activation = 'relu'),
    layers.Dense(6, activation = 'softmax'),
])
model_cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_cnn.fit(np.array([x[0] for x in x_train]), np.array([x[1] for x in x_train]), epochs = 100)
model_cnn.save('Recognition_model.h5')

# saved_model = models.load_model('Recognition_model.h5')
# prediction = saved_model.predict(np.array([x_test[1][1]]))

# print("Predicted label:", prediction)