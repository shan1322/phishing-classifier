import numpy as np
from keras.models import load_model

fea = np.load("../matpre/featest.npy")
label = np.load("../matpre/labeltest.npy")
model = load_model("../models/feadense.h5")
pred = model.evaluate(x=fea, y=label, verbose=1)
print("accuracy", pred[1] * 100)
print("loss", pred[0])
