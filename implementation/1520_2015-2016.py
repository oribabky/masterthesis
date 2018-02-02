# Load libraries
import pandas
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

from sklearn.linear_model import LinearRegression
#own library
import customlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
#xl = pd.ExcelFile("../data/stations_e6/mätvärden 1520 2015-2016.xlsx")
dataset = pandas.read_excel(open('../data/stations_e6/mätvärden 1520 2015-2016.xlsx','rb'), sheet_name='Blad1', skiprows=[0], header=1, 
    names=['År', 'Tidpunkt', 'Yttemp-MS4', 'Nedtyp-MS4', 'Ned mängd-MS4', 'Yttemp-DST111', 'Friktion-DSC111', 'Ytstatus-DSC111'],
    #dtype={'a':np.int32, 'b':np.int32, 'c':np.float64, 'd':np.int32, 'e':np.float64, 'f':np.float64, 'g':np.float64, 'h':np.int32}
    dtype=object
    )
#names = År Tidpunkt    Yttemp-MS4  Nedtyp-MS4  Ned mängd-MS4   Yttemp-DST111   Friktion-DSC111 Ytstatus-DSC111



# shape
#print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('Nedtyp-MS4').size())

#print(dataset.values)
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()


# histograms
#dataset.hist()
#plt.show()

# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()
def runModelsRegression():
    array = dataset.values
    X = array[:,3:]
    #X = customlib.sliceSkip2d(array, [0,1,2])
    Y = array[:,2]

    #split the data into training/test data
    percentageTrain = 51
    X_split = customlib.splitData(percentageTrain, X)
    X_train = X_split[0]
    X_test = X_split[1]

    Y_split = customlib.splitData(percentageTrain, Y)
    Y_train = Y_split[0]
    Y_test = Y_split[1]

    array_split = customlib.splitData(percentageTrain, array)
    array_train = array_split[0]
    array_test = array_split[1]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # Make predictions using the testing set
    Y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(Y_test, Y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(Y_test, Y_pred))

    # Plot outputs
    print(array_test[:5,1])
    print(X_test[:5,1])
    plt.scatter(array_test[:,1], Y_test,  color='black')
    plt.plot(array_test[:,1], Y_pred, color='blue', linewidth=3)

    #uncomment to hide graph value information
    #plt.xticks(())
    #plt.yticks(())

    plt.show()

def runModelsClassification():
    # Split-out validation dataset
    array = dataset.values
    X = customlib.sliceSkip2d(array, [2])
    Y = array[:,2]
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
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

#runModels()
runModelsRegression()