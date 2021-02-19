# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set
    
    unigram = []
    pos_count = Counter()
    neg_count = Counter()
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            pos_count.update(train_set[i])
        else:
            neg_count.update(train_set[i])
    
    pos_total = sum(pos_count.values())
    neg_total = sum(neg_count.values())
    
    distinct_pos = len(list(pos_count))
    distinct_neg = len(list(neg_count))
    

    for line in dev_set:
        total_pos_laplace = 0
        total_neg_laplace = 0
        for word in line:
            if pos_count[word] == 0:
                distinct_pos += 1
                distinct_neg += 1
            pos_laplace = (pos_count[word] + smoothing_parameter) / (pos_total + smoothing_parameter * (distinct_pos + 1))
            neg_laplace = (neg_count[word] + smoothing_parameter) / (neg_total + smoothing_parameter * (distinct_neg + 1))
            total_pos_laplace += math.log(pos_laplace)
            total_neg_laplace += math.log(neg_laplace)
        if total_pos_laplace + math.log(pos_prior) > total_neg_laplace + math.log(1 - pos_prior):
            unigram.append(1)
        else:
            unigram.append(0)
    return unigram


def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=1.0, bigram_lambda=0.5,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    bigram = []
    
    uni_pos_count = Counter()
    uni_neg_count = Counter()
    bi_pos_count = Counter()
    bi_neg_count = Counter()
    
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            uni_pos_count.update(train_set[i])
            bi = []
            for j in range(len(train_labels[i]) - 1):
                bi.append( train_set[i][j] + " " + train_set[i][j+1]) #matches with every word after this word
            bi_pos_count.update(bi)
        else:
            uni_neg_count.update(train_set[i])
            bi = []
            for j in range(len(train_labels[i]) - 1):
                bi.append(train_set[i][j] + " " + train_set[i][j+1])
            bi_neg_count.update(bi)
    
    uni_pos_total = sum(uni_pos_count.values())
    uni_neg_total = sum(uni_neg_count.values())
    bi_pos_total = sum(bi_pos_count.values())
    bi_neg_total = sum(bi_neg_count.values())
    
    uni_distinct_pos = len(list(uni_pos_count))
    uni_distinct_neg = len(list(uni_neg_count))
    bi_distinct_pos = len(list(bi_pos_count))
    bi_distinct_neg = len(list(bi_neg_count))
    
    for line in dev_set:
        total_uni_pos_laplace = 0
        total_uni_neg_laplace = 0
        total_bi_pos_laplace = 0
        total_bi_neg_laplace = 0
        
        for word in line:
            if uni_pos_count[word] == 0:
                uni_distinct_pos += 1
                uni_distinct_neg += 1
            
            uni_pos_laplace = (uni_pos_count[word] + unigram_smoothing_parameter) / (uni_pos_total + unigram_smoothing_parameter * (uni_distinct_pos + 1))
            uni_neg_laplace = (uni_neg_count[word] + unigram_smoothing_parameter) / (uni_neg_total + unigram_smoothing_parameter * (uni_distinct_neg + 1))
            
            total_uni_pos_laplace += math.log(uni_pos_laplace)
            total_uni_neg_laplace += math.log(uni_neg_laplace)

            
        if unigram_model_pos + bigram_model_pos > unigram_model_neg + bigram_model_neg:
            bigram.append(1)
        else:
            bigram.append(0)
            
    print(bigram)
    return bigram
    


      
