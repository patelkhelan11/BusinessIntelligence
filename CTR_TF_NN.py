import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from mlxtend.preprocessing import one_hot
from sklearn import *
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
import sys
from pandas import *
from numpy import *
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt


# This preprocessing code is not created by me ----
def labelEncode(col_names, df):
    # Convert String lables to numeric
    le = preprocessing.LabelEncoder()
    for col in col_names:
        le.fit(np.array(df[col]))
        df[col] = le.transform(np.array(df[col]))
    return df

def readData():
    # Read the train datafile as Pandas Dataframe
    df = pd.read_csv('./train.csv')
    x, y = df.drop('click', 1), df['click']


    train_df, test_df, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=42)
    train_df = train_df.append(test_df)
    return train_df, len(test_df), train_y, test_y


def removecols(df):
    # Remove columns which has high number of unique values i.e. ID
    for col in list(df.columns.values):
        if (len(df) * 0.95) <= len(df[col].unique()):
            df = df.drop(col, 1)
    return df

def changeHourCol(df):
    # Change Hour format to day, hour and date
    df['date'] = df['hour'].apply(lambda x: x%10000/100)
    df['day_hour'] = df['hour'].apply(lambda x: x%100)
    df['dow'] = df['hour'].apply(lambda x: datetime.datetime.strptime(str(((x - x%100)/100) + 20000000), '%Y%m%d').strftime('%u'))

    df = df.drop('hour', 1)
    return df


print "Preprocessing the data..."
df, test_len, train_Y, test_Y = readData()

df = changeHourCol(df)
col_names_encode_list = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                             'device_id', 'device_model', 'device_ip']

df = labelEncode(col_names_encode_list, df)
df = removecols(df)

train_data = df[1:(len(df) - test_len + 1)]
test_data = df[(len(train_data)):]

###########################################################################################################
# ===============   PRE-PROCESSING COMPLETE ===============
###########################################################################################################






print "Hello World"

#---Below code is developed by US---------------
###########################################################################################################
# === Darshak Harisinh Bhatti (dbhatti)
# === Khelan Patel (kjpatel4)
# === Sonal Patil (sspatil4)
###########################################################################################################


xx_train = train_data

# Convert Class Labels to np.Array
yy_train = array(train_Y)

# Convert Class Labels to One-Hot Labels
yy_train_onehot = one_hot(yy_train, num_labels=2, dtype='int')



xx_test = test_data

# Convert Class Labels to np.Array
yy_test = array(test_Y)

# Convert Class Labels to One-Hot Labels
yy_test_onehot = one_hot(yy_test, num_labels=2, dtype='int')


# Number of features in the first layer of neural net (input layer)
AA = len(xx_train.columns)

# Number of classes
BB = 2


#First Hidden Layer with 35 Neurons
tf_in = tf.placeholder("float", [None, AA]) # Features
tf_weight = tf.Variable(tf.zeros([AA,35]))
tf_bias = tf.Variable(tf.zeros([35]))

# Using "softmax" activation function as the task is of binary classifcation
tf_softmax_pre = tf.nn.softmax(tf.matmul(tf_in,tf_weight) + tf_bias)
print "tf_softmax_pre : ", tf_softmax_pre



# Second Hidden Layer
tf_in1 = tf.placeholder("float", [None, 35]) # Features
tf_weight1 = tf.Variable(tf.zeros([35,35]))
tf_bias1 = tf.Variable(tf.zeros([35]))
tf_softmax_pre2 = tf.nn.softmax(tf.matmul(tf_softmax_pre,tf_weight1) + tf_bias1)

print "tf_softmax_pre2 : ", tf_softmax_pre2


# Final Layer
tf_in2 = tf.placeholder("float", [None, 35]) # Features
tf_weight2 = tf.Variable(tf.zeros([35,BB]))
tf_bias2 = tf.Variable(tf.zeros([BB]))
tf_softmax = tf.nn.softmax(tf.matmul(tf_softmax_pre2,tf_weight2) + tf_bias2)
print "tf_softmax : ", tf_softmax

# Training via backpropagation
tf_softmax_correct = tf.placeholder("float", [None,BB])
#tf_cross_entropy = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))
#cost = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))




# As the given dataset has Class Imbalance
# We have used weighted cross entropy loss function
ratio = 0.05
class_weight = tf.constant([ratio, 1-ratio])
weighted_logits = tf.multiply(tf_softmax, class_weight) + tf.constant(value=1e-5)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=weighted_logits, labels=tf_softmax_correct))

tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Metrics
tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(tf_softmax_correct,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))


# Initialize Tensorflow session and run
sess = tf.Session()

# Initialize variables
sess.run(tf.global_variables_initializer())


# Start Training
for i in range(100):
    sess.run(tf_train_step, feed_dict={tf_in: xx_train, tf_softmax_correct: yy_train_onehot})
    # Calculate Metrics after every 10 iterations
    if (i%10 == 0):
        result, y_p, y_prob = sess.run([tf_accuracy, tf.argmax(tf_softmax,1), tf_softmax], feed_dict={tf_in: xx_test, tf_softmax_correct: yy_test_onehot})
        print "Log_loss", metrics.log_loss(yy_test, y_prob[0:: 1])
        rmse = sqrt(mean_squared_error(yy_test, y_prob[0:,1]))
        print "RMSE : ", rmse



