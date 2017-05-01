Click-Through Rate prediction

Readme file with instructions to execute

How to obtain data-set : This was part of a Kaggle competition. Dataset can be obtained from here
https://www.kaggle.com/c/avazu-ctr-prediction/data


1) All the coding files can be found under the “Code” folder. There will be total 4 Python files inside this folder. Please, make sure that all the files under the same directory (dataset files generated and python modules). Since, each of the file was created separately by each team member, please edit the path names inside file before executing.
Language used : Python. 
Tool used : Spyder

2) Required modules
* Pandas (pandas)
* Scikit learn (sklearn)
* Tensorflow (tensorflow)
* Sys
* Numpy (numpy)
* Datetime (datetime)
* Scikit Learn Metrics (sklearn.metrics)
* MlxTend Preprocessing (mlxtend.preprocessing)
* from sklearn import * (rmse, confusion matrix, model_selection, knn etc.)
* from numpy import * 

2) Dataset_creation.py : 
Author : Khelan Patel, Darshak Bhatti, Sonal Patil
This file converts the original “TextEdit” document of dataset (as downloaded from Kaggle) to the appropriate CSV file format which can be used to load data as a panda dataframe in Python.
Output file name : CTR_dataset.csv
Input file name : train

3) Logistic_CTR.py
Author : Khelan Patel (kjpatel4@ncsu.edu)
Description : This file implements generalized linear models (logistic regression model and logistic regression model with SGD and L2 regularization. Parameter values alpha=0.00025, loss="log", penalty="l2"). Fits the model using 67% training data and tests it on the validation set (33%)
Output : For the evaluation purpose, it prints the logistic loss or cross-entropy loss of the model. Defined as the negative log-likelihood of the true labels given a probabilistic classifier’s predictions (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).

4) CTR_TF_NN.py
Author : Darshak Bhatti (dbhatti@ncsu.edu)
Description : This code implements construction and training of Neural Network with Tensorflow library. It is a 2-hidden layer neural network with Softmax activation function and Class-Weighted Cross Entropy Loss function to minimize. It uses Gradient Descent with learning rate 10e-2
Output : For the evaluation purpose, it prints the logistic loss or cross-entropy loss of the model. Defined as the negative log-likelihood of the true labels given a probabilistic classifier’s predictions (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).

5) KNN_CTR.py
Author : Sonal Patil (sspatil4@ncsu.edu)
Description : This code implements K-Nearest Neighbor classification on the dataset. The model was built on a training-test split of 60:40.
Output : For the evaluation purpose, it prints the logistic loss or cross-entropy loss of the model. Defined as the negative log-likelihood of the true labels given a probabilistic classifier’s predictions (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
