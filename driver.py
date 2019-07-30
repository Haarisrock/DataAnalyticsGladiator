from slicing_data import slicing
from fitknn import knn
from fitsvm import svm
from fitbsn import bsn
from fitdtc import dtc

ls=slicing('nba_logreg.csv',1)  #Slicing the dataset and storing it in a list ls
print('NBA')

#Passing the sliced data to each model
knn(ls)
svm(ls,'linear')
dtc(ls)
bsn(ls)

print('----------------------------------------------------------------------')

ls=slicing('diabetes.csv',0)
print('DIABETES')
knn(ls)
svm(ls,'rbf')
dtc(ls)
bsn(ls)

print('----------------------------------------------------------------------')

ls=slicing('spambase.csv',0)
print('Spambase')
knn(ls)
svm(ls,'rbf')
dtc(ls)
bsn(ls)

print('----------------------------------------------------------------------')




ls=slicing('mammographic_massesdata.csv',0)
print('MGM')
knn(ls)
svm(ls,'rbf')
dtc(ls)
bsn(ls)

print('----------------------------------------------------------------------')



