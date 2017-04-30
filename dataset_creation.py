#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:45:03 2017

@author: khelanpatel
"""

import random
import pandas

#input file name: Downloaded from kaggle
input_file = "train"
#output file name : Saves below name
output_file = "ctr_dataset.csv"
input_len = sum(1 for line in open(input_file)) - 1 
print input_len
lines = 5000000
skip = sorted(random.sample(range(1, input_len+1),input_len-lines))

#generates panda dataframe from TextEdit file
df = pandas.read_csv(input_file,skiprows = skip)

#stores panda data-frame as CSV file
df.to_csv(output_file)