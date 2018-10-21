import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout
from keras.models import Sequential

x_train = np.load("../matpre/embtrain.npy")
label = np.load("../matpre/latrain.npy")
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

print(x_train.shape)
print(label.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, label, batch_size=64, epochs=1, verbose=1)
model.save('../models/CNN.h5')
