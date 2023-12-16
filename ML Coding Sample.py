#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code sample examines how the zero-one loss of a decision tree classifier can vary with the maximum number of leaf nodes.
Please make sure to download the "training and test data" folder on my Github and set your working directory to the folder containing the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

#4i

training_data = np.load('train.npy')
test_data = np.load('test.npy')
trainlabels = np.load('trainlabels.npy')
testlabels = np.load('testlabels.npy')

for i in range(0, 10):
    plt.imshow(training_data[i], cmap='gray')

#4ii

#training_data needs to be reshaped

x_train = np.reshape(training_data, (training_data.shape[0], -1))
x_test = np.reshape(test_data, (test_data.shape[0], -1))


# Source: https://www.datacamp.com/tutorial/decision-tree-classification-python

max_leaf_nodes = [2, 8, 32, 128, 256, 512, 1024, 2048, 4096, 5500]
train_loss_li = []
test_loss_li = []

for num in max_leaf_nodes:
    classifier = DecisionTreeClassifier(max_leaf_nodes = num)
    clf_train = classifier.fit(x_train, trainlabels)
    
    y_pred_test = clf_train.predict(x_test)
    y_pred_train = clf_train.predict(x_train)
    
    test_loss = zero_one_loss(testlabels, y_pred_test)
    test_loss_li.append(test_loss)
    
    train_loss = zero_one_loss(trainlabels, y_pred_train)
    train_loss_li.append(train_loss)

loss_plot = plt.figure()
plt.plot(max_leaf_nodes, train_loss_li, label="Training data loss")
plt.plot(max_leaf_nodes, test_loss_li, label="Test data loss")
plt.xscale("log")
plt.xlabel("Maximum Number of Leaf Nodes (log)")
plt.ylabel("Zero-One Loss")    
plt.title('Plot of Zero-One Loss against Maximum Number of Leaf Nodes')
plt.legend()
plt.show()

print(min(train_loss_li))
print(min(test_loss_li))

