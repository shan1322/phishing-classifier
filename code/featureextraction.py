from collections import Counter
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/urldata.csv')
label1 = np.array(df['label'])
url = np.array(df['url'])


# function converts bad to 0 and good to 1
def label_encode(label):
    enclabel = []
    for i in label:
        if i == "bad":
            enclabel.append(0)
        else:
            enclabel.append(1)
    return np.array(enclabel)


# freq of words
def word_count(url):
    words = []
    for i in url:
        parse = urlparse(i)
        path = parse.path
        path = path.split('.')
        words.append(path[0])
    count = Counter(words)
    freq = count.values()
    return np.array(freq)


# length of urls
def lenurl(url):
    lenurl = []
    for i in url:
        lenurl.append(len(i))
    return np.array(lenurl)


# count no of digits in url
def digits(url):
    num = []
    for i in url:
        numbers = sum(c.isdigit() for c in i)
        num.append(numbers)
    return np.array(num)


# no of dots in url
def dots(url):
    dot = []
    for i in url:
        count = Counter(i)
        dot.append(count['.'])
    return np.array(dot)


lu, digit, do = lenurl(url), digits(url), dots(url)
features = []
label = label_encode(label1)
laa = label
for i in range(0, len(lu)):
    temp = []
    temp.append(lu[i])
    temp.append(digit[i])
    temp.append(do[i])
    features.append(temp)
label = np_utils.to_categorical(label, 2)
fea_train, fea_test, label_train, label_test = train_test_split(features, label, test_size=0.20, random_state=42)
np.save("../matpre/featrain.npy", fea_train)
np.save("../matpre/featest.npy", fea_test)
np.save("../matpre/labeltrain.npy", label_train)
np.save("../matpre/labeltest.npy", label_test)

### plot
plt.plot(laa, do, color='pink')
plt.xlabel('label')
plt.ylabel('no of dots')
plt.show()
