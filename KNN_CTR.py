import pandas as pd
import sys
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from math import sqrt

#Process data from text file into csv file for usage in code
def process_csv(filename):
	fname = str(filename)
	outputfile = "/home/sonal/BI_project/" + f + "_5million.csv"
	len_df = sum(1 for line in open(fname)) - 1
	lines = 5000000
	skip = sorted(random.sample(range(1, len_df+1),len_df-lines))
	df = pd.read_csv(fname,skiprows = skip)
	return df

#Convert to discrete features by using label encoding converting all categorical features to numeric
def convertToDiscrete(data):
	contFeatures = []
	for col in list(data):
		le = preprocessing.LabelEncoder()
		if data[str(col)].dtype == "object":
			contFeatures.append(col)
			le.fit(np.array(data[col]))
			data[col] = le.transform(np.array(data[col]))

	return data, contFeatures

#Using PCA to reduce dimensionality of the components to top 20 and convert them into feature set for processing
def pcaFeatures(data, test_df):
	pca = PCA(n_components = 12)
	data_new = pca.fit_transform(data)
	data_new = pd.DataFrame(data_new)
	featureNames=[]
	for i in range(12):
		featureNames.append("f"+str(i))
	data_new.columns=featureNames
	for i in range(12):
		data_ = data_new["f"+str(i)]

	#print data_.shape

	test_new = pca.transform(test_df)
	test_new = pd.DataFrame(test_new)
	featureNames=[]
	for i in range(12):
		featureNames.append("f"+str(i))
	test_new.columns=featureNames
	for i in range(12):
		test_ = test_new["f"+str(i)]

	#print test_.shape


	return data_, test_


#To build a K Nearest Neighbors classifier where classification is computed from a simple majority vote of the nearest neighbors of each point:

def KNNModel(train_df, train_y, test_X, test_y):
    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(train_df, train_y)
    predicted = model.predict(test_X)
    acc = accuracy_score(predicted, test_y)
    a = model.predict_proba(test_X)
    print "Log loss:", log_loss(test_y, a)
    print "RMSE:", sqrt(mean_squared_error(test_X, predicted[0:: 1]))
    return acc, model

if __name__ == "__main__":
	#train_df = process_csv("train")
	train_df = pd.read_csv("/home/sonal/BI_project/train_5million.csv")
	
	train = train_df.sample(frac=0.6, random_state=100)
	test = train_df.drop(train.index)
	print "1"
	train_X, train_y = train.drop(['click', 'id', 'device_id'], 1), train['click']
	print "here"
	test_X, test_y = test.drop(['click', 'id', 'device_id'], 1), test['click']

	train_df, contFeatures = convertToDiscrete(train_X)
	test_df, contFeatures = convertToDiscrete(test_X)
	
	train_X, test_X = pcaFeatures(train_df, test_df)
	
	train_X = train_X.reshape(-1,1)
	test_X = test_X.reshape(-1,1)
	
	
	acc = KNNModel(train_X, train_y, test_X, test_y)
	print "KNN acc", acc
