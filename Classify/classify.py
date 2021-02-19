# classify.py
# Submitted by: Sooryaprakash Raja Venkataramanan
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    W = np.zeros(len(train_set[0]))
    b = 0
    #print(train_set[:, 0], len(train_set[:, 0]))
    for i in range(max_iter):
        for j in range(len(train_set)):
            pred = np.sign(np.dot(W, train_set[j]) + b)
            if pred == 1:
                if train_labels[j] != 1:
                    W -= train_set[j] * learning_rate
                    b -= learning_rate
            else:
                if train_labels[j] != 0:
                    W += train_set[j] * learning_rate
                    b += learning_rate
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    labels = []
    for x in dev_set:
        pred = np.sign(np.dot(W, x) + b)
        if pred == 1:
            labels.append(1)
        else:
            labels.append(0)
    return labels

import math
import heapq
from collections import Counter

def euclidean(test, train):
    return np.linalg.norm(test-train)
    
def closest_neighbors(test, train_set, train_labels, k):
    diff = []
    for i in range(len(train_set) - 1):
        dist = euclidean(test, train_set[i])
        heapq.heappush(diff, (dist, train_labels[i]))
    neighbors = []
    for i in range(k):
        neighbors.append(heapq.heappop(diff)[1])
    return neighbors
    
def classifyKNN(train_set, train_labels, dev_set, k):
    pred = []
    for test in dev_set:
        neighbors = closest_neighbors(test, train_set, train_labels, k)
        common = np.argmax(np.bincount(neighbors))
        pred.append(common)
    return pred
