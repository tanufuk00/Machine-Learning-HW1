# -*- coding: utf-8 -*-
#-----------------------------------------------------------
"""
Tan Ufuk Çelik

Original file is located at
    https://colab.research.google.com/drive/1P3jErd6hJMW6qQ-7JhS60BA8jZh8SrUn

# CS412 - Machine Learning Course

## Software:

You may find the necessary function references here:

http://scikit-learn.org/stable/supervised_learning.html

"""

#-----------------------------------------------------------

# import random and numpy libraries.
import random
import numpy as np

# To remain always same random mix.
random.seed(412)
np.random.seed(412)

#-----------------------------------------------------------

## 2) Load training dataset

#Read MNIST dataset from Keras library.
from keras.datasets import mnist

# First, I will divide the dataset as train and test sets. 
# Later, I will divide the train set as the development and validation set.
(X_train, Y_train), (X_test, y_test) = mnist.load_data()

#-----------------------------------------------------------

#3) Reshape the MNIST data
#In order to use images as input data for the sklearn k-NN classifier, the 2D image arrays need to be reshaped into a 1D arrays (in other words, a feature vector).
print('Before reshaping: ', X_train.shape, X_test.shape)

# For example; N_train, height, width = (a, b, c)
N_train, height, width = X_train.shape

X_train = np.reshape(X_train, (N_train, height*width)) # from array of shape N_train x 28 x 28 ---> N_train x 784 (Note: 28*28 = 784)
X_test = np.reshape(X_test, (len(X_test), height*width)) # from array of shape N_test x 28 x 28 ---> N_test x 784

print('After reshaping: ', X_train.shape, X_test.shape)

#-----------------------------------------------------------

#4) Shuffle and Split TRAINING data as train (also called development) (80%) and validation (20%)

from sklearn.utils import shuffle

# Shuffle the training data
X_train, Y_train = shuffle(X_train, Y_train)

# Split %80 train - 20% val
X_train_size = int(0.8 * len(X_train))
Y_train_size = int(0.8 * len(Y_train))

#Train (%80)
X_dev = X_train[:X_train_size]
Y_dev = Y_train[:Y_train_size]

#Validation (%20)
X_val = X_train[X_train_size:]
Y_val = Y_train[Y_train_size:]

#-----------------------------------------------------------

#5) Train k-NN  classifier on development data and do model selection using the validation data
#* Train a k-NN classifier (use the values specified in the homework PDF file, do not try other values) with the rest of the parameters set to default.
#* The aim in this homework is not necessarily obtaining the best performance, but to establish the ML pipeline (train a few models, select based on validation set, test, report).


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_values = [1, 3, 5, 7, 9, 11, 13]   # <--- Fill the list with the values for n_neighbors

best_acc = -1
best_k = None
val_accs = []
for k in k_values:
  # 1) initialize a k-NN classifier with n_neighbors parameter set to k
  # 2) train the classifier using training set
  # 3) get the predictions of the classifier on the validation set
  # 4) compute the accuracy of the predictions on the validation set and append it to val_accs list

  #1
  classifier = KNeighborsClassifier(n_neighbors=k)
  #2
  classifier.fit(X_dev, Y_dev)
  #3
  Y_pred = classifier.predict(X_val)
  #4
  accuracy_val_set = accuracy_score(Y_val, Y_pred)
  val_accs.append(accuracy_val_set)

  print('Validation accuracy for k=', k, ' :', accuracy_val_set, 'your validation accuracy')
  # if validation accuracy is better than best_acc, update best_acc and best_k

#I took the max value and assigned it to the variable best_acc.
best_acc = max(val_accs)
#I found the index of the string
index_accs = val_accs.index(best_acc)
#I assigned it to the best_k variable from the index.
best_k = k_values[index_accs]

print('Best validation accuracy (', best_acc, ') is achieved with k=', best_k)

#-----------------------------------------------------------

# 6) Plot the obtained validation accuracies versus k values"""

import matplotlib.pyplot as plt

plt.plot(k_values, val_accs)
plt.xticks(k_values)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.show()

#-----------------------------------------------------------

# 7) Test your classifier on test set

#Now that you have the best value for the ***n_neighbors*** parameter, train a model **with best parameters that you have found according to your validation results**. But now, train the model by combining the training and validation sets. Then report the accuracy on the test set.

# 1) initialize a k-NN classifier with n_neighbors parameter set to best_k
# 2) combine the training and validation sets (you may want to look up numpy.concatenate function for this)
# 3) train the classifier using this set
# 4) get the predictions of the classifier on the test set
# 5) compute the accuracy of the predictions on the test set
#print('Test accuracy for k=', best_k, ' :', 'your test accuracy')

#1- For best_k, k-NN classifier is crated.
classifier_best_k = KNeighborsClassifier(n_neighbors=best_k)

#2- combine the training and validation sets
#The -1 parameter in the reshape(-1,1) function allows numpy to automatically calculate the size.
#The 1 parameter specifies that the new shape will be 1 dimensional.

# X_dev ve X_val conc.
X_merged = np.concatenate((X_dev, X_val), axis=0)

# Y_dev ve Y_val'i conc.
Y_merged = np.concatenate((Y_dev, Y_val), axis=0)

#3- train the classifier using the set.
classifier.fit(X_merged, Y_merged)

#4-get the predictions of the classifier on the test set
Y_pred2 = classifier.predict(X_test)

#5- compute the accuracy of the predictions on the test set
accuracy_test_set = accuracy_score(y_test, Y_pred2)

print('Test accuracy for k=', best_k, ' :', accuracy_test_set, 'your test accuracy')

#-----------------------------------------------------------

# Report your result

# I added my report as pdf file.