# !/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *

train_path = r'..\training2.txt'
test_path = r'..\test2.txt'
parameter = '-t 2 -c 2 -g 0.5'
train_label, train_value = svm_read_problem(train_path)
test_label, test_value = svm_read_problem(test_path)

model = svm_train(train_label, train_value, parameter)
p_label, p_acc, p_val = svm_predict(test_label, test_value, model)

print(p_acc)
