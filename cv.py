#load the data
from sklearn import datasets, svm
iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target

#create the folds and calculate the scores
import numpy as np
svc = svm.SVC(C=1, kernel='linear')
X_folds = np.array_split(X, 5)
y_folds = np.array_split(y,5)
scores = list() #create an empty list
for k in range(5):
	X_train = list(X_folds) #copy X-folds to x-train
	X_test = X_train.pop(k) #create the x-test array by popping a part of x-train
	X_train = np.concatenate(X_train) #make a single array. Thus, x = [[1,2],[3,4]] becomes [1,2,3,4]
	y_train = list(y_folds)
	y_test = y_train.pop(k)
	y_train = np.concatenate(y_train)
	scores.append(svc.fit(X_train,y_train).score(X_test, y_test))

print(scores)

#calculate the mean of the scores
def avg(list):
	sum = 0
	for elm in list:
		sum += elm
	print("The mean of the scores is: " + str(sum/(len(list)*1.0))) #it was multiplied with 1.0 to make the int into a double

avg(scores)
print(np.mean(scores)) #another method without writing a function
print(np.std(scores))

'''The mean of the scores is: 0.61333333333 and standard deviation is 0.308832928584'''


'''The above code is written without using the cross-validation module of scikit. So, below is the code written with cross-validation.'''

from sklearn import cross_validation
kfold = cross_validation.KFold(len(X), n_folds=5)
[svc.fit(X[train], y[train]).score(X[test], y[test]) for train, test in kfold]

''' same scores were obtained in both the methods: [1.0, 0.80000000000000004, 0.29999999999999999, 0.76666666666666672, 0.20000000000000001]'''


#if we want to choose a different method for the score-estimator, then we use the following:
cross_validation.cross_val_score(svc, X, y, cv=kfold, n_jobs=-1) #n_jobs is #of CPUs to use; -1 means all CPUs

'''we obtain same score again -- array([ 1.        ,  0.8       ,  0.3       ,  0.76666667,  0.2       ]). Thus, the above 1-line code is much easier to use to compute the k-fold scores. Also, the code gives us the opportunity to choose the estimator method we want --  f1, r2, recall, etc, etc.'''


#f1, r2 and recall score
cross_validation.cross_val_score(svc, X, y, cv=kfold, scoring='f1', n_jobs=-1) #n_jobs is #of CPUs to use; -1 means all CPUs
cross_validation.cross_val_score(svc, X, y, cv=kfold, scoring='r2', n_jobs=-1) 
cross_validation.cross_val_score(svc, X, y, cv=kfold, scoring='recall', n_jobs=-1) 


'''The score for f1 is array([ 0.        ,  0.85714286,  0.46153846,  0.74074074,  0.        ]). The score for r2 is array([ 1.  ,  0.1 ,  0.  , -0.05,  0.  ]). The score for recall is array([ 0. ,  0.8,  0.3,  1. ,  0. ]).'''
