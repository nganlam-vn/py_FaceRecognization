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
    layers.Dense(3, activation = 'softmax'),
])
model_cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_cnn.fit(np.array([x[0] for x in x_train]), np.array([x[1] for x in x_train]), epochs = 100)
model_cnn.save('Recognition_model.h5')
