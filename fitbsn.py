from pomegranate import *
from sklearn.metrics import confusion_matrix,mean_squared_error

def bsn(dt):

	#fitting the Data into the model BayesisClassifier in clf
	clf=BayesClassifier.from_samples(MultivariateGaussianDistribution,dt[0],dt[1])  #dt[0] is the xtrain, dt[1] is ytrain
	
	#Predicting xtest
	pred=clf.predict(dt[2])  #dt[2] is xtest
	pred=pred.ravel()
	
	#Reshaping the data 
	dt[3]=dt[3].reshape(-1,1)
	pred=pred.reshape(-1,1)
	
	#Creating and Storing the confusion matrix and printing it
	cm = confusion_matrix(dt[3],pred)   #dt[3] is ytest
	print('Bayesian Classifier')
	print(cm)
	
	#Finding the mean squared error and printing it
	error=mean_squared_error(dt[3],pred)
	print('Mean Squared Error= ' "%.2f" % round(error*100,2) , '%')
	print()
