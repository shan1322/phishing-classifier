import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras_preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('../data/urldata.csv')
url = np.array(df['url'])
label1 = np.array(df['label'])
siz = int(0.1 * len(label1))
url = url[0:siz]


# no of dots in url
def label_encode(label):
    enclabel = []
    for i in label:
        if i == "bad":
            enclabel.append(0)
        else:
            enclabel.append(1)
    return enclabel


# one hot encode
encoded_docs = [one_hot(d, 20 * len(url)) for d in url]
leng = []
for i in encoded_docs:
    leng.append(len(i))
print(max(leng))
padded_docs = pad_sequences(encoded_docs, maxlen=max(leng), padding='post')

label = label_encode(label1)
label = label[0:siz]
la = label
label = np_utils.to_categorical(label, 2)

model = Sequential()
model.add(Embedding(siz, 32, input_length=max(leng)))
input_array = padded_docs
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
x, y = [], []
for i in output_array:
    pca = PCA(n_components=2)
    a = pca.fit(i)
    a, b = pca.singular_values_
    x.append(a)
    y.append(b)

fig = plt.figure()
ax = Axes3D(fig)
print(label.shape)
ax.scatter(x, y, la)
plt.xlabel('xcomponet')
plt.ylabel('ycomponet')
# plt.zlabel('label')


plt.show()

out_train, out_test, label_train, label_test = train_test_split(output_array, label, test_size=0.05, random_state=42)
np.save("../matpre/embtrain.npy", out_train)
np.save("../matpre/embtest.npy", out_test)
np.save("../matpre/latrain.npy", label_train)
np.save("../matpre/latest.npy", label_test)
