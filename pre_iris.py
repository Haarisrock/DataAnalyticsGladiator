import csv
file1 = open('iris.csv','r')
inputfile= csv.reader(file1)

new_row_list=[]


for row in inputfile:
	try:
		if row[4]=='Iris-setosa':
			p=0
		elif row[4]=='Iris-versicolor':
			p=1
		elif row[4]=='Iris-virginica':
			p=2
		new_row=[row[0],row[1],row[2],row[3],p]
		new_row_list.append(new_row)
	except Exception:
		print(row)
file1.close()

import numpy as np
np.savetxt('iris5.csv', new_row_list, delimiter=',', fmt='%s')




