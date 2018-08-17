import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from PIL import Image
from sklearn.cluster import KMeans


os.chdir("/media/adrian/Files/Documents/MCS-DS/AppliedMachineLearning_CS498/HW7CS498AML")

file1 = "docword.nips.txt"

file2 = "vocab.nips.txt"

with open(file1) as f:
    data = f.read().split("\n")
with open(file2) as g:
    vocab = g.read().split("\n")

data1 = data[3:len(data) - 1]

# pre-allocate an array of zeros (1500 docs, 12419 words)
dt = np.zeros((1500, 12419))
# create data matrix [docID, wordID]
for i in range(np.shape(data1)[0]):
    docID = int(data1[i].split()[0]) - 1
    wordID = int(data1[i].split()[1]) - 1
    count = float(data1[i].split()[2])

    dt[docID, wordID] = count

W = np.zeros((1500, 30))

topic = 30
pi_j = np.zeros(topic)

word = np.shape(dt)[1]  # 12419

document = np.shape(dt)[0]  # 1500

pi_j = np.random.uniform(low=0.0, high=1.0, size=(30,))
p_jk = np.random.uniform(low=0.0, high=1.0, size=(topic,np.shape(dt)[1]))

W1_nom = np.zeros((document, topic))
temp1 = np.zeros((document, topic))
Z = np.zeros((document, np.shape(dt)[1]))

#### E-step ####
QQ1 = []
QQ1.append(0)
for k in range(1000):
    for i in range(document):
        for j in range(topic):
            temp1[i, j] = np.sum(dt[i,] * np.log(p_jk[j,])) + np.log(pi_j[j])

        Max = temp1.max(1)
        temp2 = np.exp(temp1 - Max[::, None])
        W = temp2 / temp2.sum(1)[::, None]

    ####M-step####
    p_jk_nom = np.zeros((topic, dt.shape[1]))
    for i in range(topic):
        for j in range(document):
            p_jk_nom[i,] += np.dot(dt[j,], W[j, i])
    p_jk_nom += 0.00001  # prevent 0 word probability
    p_jk_denom = np.sum(p_jk_nom, axis=1) + 0.00001 * dt.shape[1]
    p_jk = p_jk_nom / p_jk_denom[::, np.newaxis]

    pi_j = np.sum(W, axis=1) / document

    ####convergence check####
    Q1 = 0
    for i in range(document):
        for j in range(topic):
            Q1 += (np.dot(dt[i,], np.log(p_jk[j, :])) +
                   np.log(pi_j[j])) * W[i, j]
    QQ1.append(Q1)

    # check convergence
    if np.abs(QQ1[k] - QQ1[k - 1]) < 0.0001:
        break

# get the probability the topic is selected
topic_index = np.zeros(document)
prob = np.zeros(30)
for i in range(document):
    topic_index[i] = np.argsort(W[i,])[::-1][0]
    for j in range(topic):
        if topic_index[i] == j:
            prob[j] += 1
prob = prob / document
ax, fig = plt.subplots()
plt.bar([i for i in range(1, 31)], prob)
plt.xlabel('Topic')
plt.ylabel('Probability')
plt.title("Topic Probability")
plt.show()
## get top 10 most frequent word from each topic

Vocab = vocab[0:len(vocab) - 1]
freq_word_index = []
freq_word = np.zeros((topic, 10))

for i in range(topic):
    freq_word_index.append(np.argsort(p_jk[i,])[::-1][0:10])

freq_word = np.zeros((30, 10), dtype=object)

for i in range(len(freq_word_index)):
    for j in range(10):
        freq_word[i][j] = vocab[freq_word_index[i][j]]

df = pd.DataFrame(freq_word)