import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

fea = np.load("../matpre/featrain.npy")
label = np.load("../matpre/labeltrain.npy")
print(fea.shape)
print(label.shape)
batch_size = 1000
nb_classes = 2


def model_dense(x_train):
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = model_dense(fea)
model.fit(fea, label, batch_size=batch_size, epochs=10, verbose=1)
model.save("../models/feadense.h5")
