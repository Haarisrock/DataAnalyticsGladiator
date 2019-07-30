import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

def slicing(fname,st):
	df= pd.read_csv(fname,header=0)  #Reading the csv file
	
	
	#Splitting the data into prdictors and responses
	y=df.iloc[:,-1:]
	x=df.iloc[:,st:-1]

	#Filling the null values with zeros
	x=x.fillna(0)
	
	#Splitting the data in training and testing
	xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
	
	#Converting the dataframe to a matrix
	ytrain=ytrain.as_matrix()
	ytest=ytest.as_matrix()
	
	#Flattening the matrix
	ytest=ytest.ravel()
	ytrain=ytrain.ravel()

	#Storing the sliced data in a list
	ls=[xtrain,ytrain,xtest,ytest]
	
	#returning the list containing sliced data
	return ls


