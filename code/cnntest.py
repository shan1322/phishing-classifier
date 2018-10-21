import numpy as np
from keras.models import load_model

x_test = np.load("../matpre/embtest.npy")
label = np.load("../matpre/latest.npy")
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
model = load_model("../modelS/CNN.h5")
pred = model.evaluate(x=x_test, y=label, batch_size=1, verbose=1)
print("accuracy", pred[1] * 100)
print("loss", pred[0])
