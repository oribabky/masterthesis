# Load libraries
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('class').size())

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

# histograms
#dataset.hist()
#plt.show()

# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))

# evaluate each model in turn
names = []

# feature extraction
#test = SelectKBest(score_func=chi2, k=2)
#fit = test.fit(X_train, Y_train)
# summarize scores
#numpy.set_printoptions(precision=3)
#print(fit.scores_)
#features = fit.transform(X)
# summarize selected features
#print(features[0:5,:])

# Make predictions on validation dataset
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    predictions_training = model.predict(X_train)
    # Plot outputs
    #plt.scatter(X_validation, Y_validation,  color='black')
    #plt.plot(X_validation, predictions, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()
    print("training performance")
    print(accuracy_score(Y_train, predictions_training))
    print("test performance")
    print(accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    #print(classification_report(Y_validation, predictions))

print("TIME TO GRID")
knn = KNeighborsClassifier()
print(accuracy_score(Y_validation, predictions))
ks = np.array([2,4,8,10])
grid = model_selection.GridSearchCV(estimator=knn, param_grid=dict(n_neighbors=ks))
grid.fit(X_train, Y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)