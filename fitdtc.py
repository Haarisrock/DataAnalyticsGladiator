from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.tree import DecisionTreeClassifier

def dtc(dt):
	dtree=DecisionTreeClassifier(max_depth=2)
	dtree.fit(dt[0],dt[1])
	pred=dtree.predict(dt[2])
	pred=pred.ravel()
	dt[3]=dt[3].reshape(-1,1)
	pred=pred.reshape(-1,1)
	cm = confusion_matrix(dt[3],pred)
	print('Decision Tree')
	print(cm)
	error=mean_squared_error(dt[3],pred)
	print('Mean Squared Error= ' "%.2f" % round(error*100,2) ,'%')
	print()
