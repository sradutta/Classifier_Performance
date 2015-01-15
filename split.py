from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import datasets

#using only sepal length and width
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

#split the dataset into training and testing set with 40% of data for testing
#random_state = Pseudo-random number generator state used for random sampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=33)
print(X_train.shape, y_train.shape) #telling us how many data-points are in x-train and y-train. Thus, knowing the size of the dataset, we can make sure that really 40% of the data is used for testing. 

'''I've 90 points in my training set; thus the remainder, 150 - 90 = 60 are in my test-set.'''

#standardize the features -- the data are normalized so that they have a mean = 0 and SD = 1
scalar = preprocessing.StandardScaler().fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#plotting to see how the training instances are distributed
import matplotlib.pyplot as plt
colors=['red', 'greenyellow', 'blue']
for i in range(len(colors)):
	xs = X_train[:,0][y_train == i] #the array value for the first-flower, i = 0 and then i = 1, i = 2
	ys = X_train[:,1][y_train == i]
	plt.scatter(xs,ys,c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

#Support Vector Machine Classifier
from sklearn import svm
svc = svm.SVC(kernel='linear')
svc.fit(X,y)
print(svc.fit(X,y).score(X,y))

#Support Vector Machine Visualization
from matplotlib.colors import ListedColormap
import numpy as np

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
plot_estimator(svc, X, y)
plt.show()








