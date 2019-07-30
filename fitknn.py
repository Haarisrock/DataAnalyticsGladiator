from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,mean_squared_error

def knn(dt):

	#Storing the model K-Neighbour Classifier in knn
	knn=KNeighborsClassifier()
	
	#fitting The Data into knn
	knn.fit(dt[0],dt[1])  #dt[0] is the xtrain, dt[1] is ytrain
	
	#Predicting xtest
	pred=knn.predict(dt[2])  #dt[2] is xtest
	
	#Reshaping the data 
	pred=pred.ravel()
	dt[3]=dt[3].reshape(-1,1)
	pred=pred.reshape(-1,1)
	
	#Creating and Storing the confusion matrix and printing it
	cm=confusion_matrix(dt[3],pred)  #dt[3] is ytest
	print('K-Nearest Neighbour')
	print(cm)
	
	#Finding the mean squared error and printing it
	error=mean_squared_error(dt[3],pred)
	print('Mean Squared Error= ' "%.2f" % round(error*100,2) ,'%')
	print()
	

