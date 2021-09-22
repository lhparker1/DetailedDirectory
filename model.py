#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:52:39 2020

@author: liamparker
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import time

class model:
    def __init__(self, train, test, train_label, test_label):
        """
        :param train->array: represents the training data set
        :param test->array: represents the validation data set
        :param train_label->list: represents the training label list
        :param test_label->list: represents the validation label list
        """
            
        self.train     = train
        self.test      = test
        self.trl       = train_label
        self.tel       = test_label
    

    @ignore_warnings(category=ConvergenceWarning)
    def logistic_regression(self):
        """Logistic regressor model from SKLearn""" 
        clf = LogisticRegressionCV(penalty = 'l2', max_iter = 500, solver = 'saga')
        clf.fit(self.train, self.trl)
        score = clf.score(self.test, self.tel)
        return clf, score

    def CNN(self):
        """Input CNN code"""


