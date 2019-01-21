from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle
import math

#get data
print('get data\n')
mnist = fetch_mldata('MNIST original')
#nomalization
print('start nomalization\n')
for item in mnist.data[:]:
	item = item/255
print('nomalization done\n')
#shuffle data
mnist.data,mnist.target=shuffle(mnist.data,mnist.target)
#Reducing dimention
print('start Reducing dimention\n')
pca = PCA(n_components=400).fit(mnist.data)
new_mnist = pca.transform(mnist.data)
print(new_mnist.shape)
split = math.ceil(len(mnist.target)*0.8)
print('end Reducing dimention\n')
#training
print('start training\n')
clf = SVC(verbose=1)
clf.fit(new_mnist[:split],mnist.target[:split])
print('start count acc\n')
expected = mnist.target[split:]
predicted = clf.predict(new_mnist[split:])
print("Classification report for classifier %s:\n%s\n"
    % (clf, metrics.classification_report(expected, predicted)))
