# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
import numpy

from sklearn.linear_model import LinearRegression
#own library
import customlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
#xl = pd.ExcelFile("../data/stations_e6/mätvärden 1520 2015-2016.xlsx")
#file = 'mätvärden 1520 2015-2016.xlsx'
file = 'mätvärden total.xlsm'
sheet = 'raw_noerr_featureengi'
dataset = pandas.read_excel(open('../data/stations_e6/' + file,'rb'), sheet_name=sheet, skiprows=[0], header=1, 
    #names=['Time', 'SurfTemp(TIRS)', 'PrecType', 'PrecAmount', 'SurfTemp(DST111)', 'Friction', 'SurfStatus'],
    #dtype={'a':np.int64, 'b':np.float64, 'c':np.int64, 'd':np.float64, 'e':np.float64, 'f':np.float64, 'g':np.int64}
    
    #when using feature engineered names
    names=['Month', 'Hour', 'SurfTemp(TIRS)', 'PrecType', 'PrecAmount', 'SurfTemp(DST111)', 'Friction', 'SurfStatus'],
    dtype={'a':np.int64, 'b':np.int64, 'c':np.float64, 'd':np.int64, 'e':np.float64, 'f':np.float64, 'g':np.float64, 'h':np.int64}
    
    )

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('PrecType').size())
#dataset.hist(column = 'PrecType')
#plt.show()



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



def modelSurfaceTemperature(skipFeatures, testSize, targetIndex, crossVal, split, featureComparison):
    array = dataset.values
    x = customlib.sliceSkip2d(array, skipFeatures)
    y = array[:,targetIndex]

    seed = 7
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=testSize, random_state=seed)

    if featureComparison:
        # feature extraction
        test = SelectKBest(score_func=f_regression, k='all')
        fit = test.fit(xTrain, yTrain)
        # summarize scores
        #numpy.set_printoptions(precision=3)
        print(fit.scores_)
        print(sorted(fit.scores_))

    # Spot Check Algorithms
    models = []
    models.append(('OLS', LinearRegression()))
    models.append(('CART', tree.DecisionTreeRegressor()))
    models.append(('kNN', KNeighborsRegressor()))
    models.append(('BP', MLPRegressor()))
    models.append(('Lasso', Lasso()))
    models.append(('RF', RandomForestRegressor()))
    
    #models.append(('NB', GaussianNB()))
    
        # evaluate each model in turn
    results = []
    names = []
    crossPlot = False
    if crossVal:
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring='neg_mean_squared_error')
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

            if crossPlot:
                #plot
                predicted = model_selection.cross_val_predict(model, x, y, cv=kfold)

                fig, ax = plt.subplots()
                ax.scatter(y, predicted, edgecolors=(0, 0, 0))
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                ax.set_xlabel('Measured')
                ax.set_ylabel('Predicted')
                plt.show()

    if split:
        for name, model in models:
            model.fit(xTrain, yTrain)
            yPred = model.predict(xTest)
            yPredTrain = model.predict(xTrain)

            print(name + ": Test MSE: %(mseTest).2f MSE diff: %(mseDiff).2f" % \
             {"mseTest": mean_squared_error(yTest, yPred), 
             "mseDiff": (mean_squared_error(yTest, yPred) - mean_squared_error(yTrain, yPredTrain))})
            #print(name + ": diff MSE: %(mseDiff).2f " % \
             #{"mseDiff": (mean_squared_error(yTest, yPred) - mean_squared_error(yTrain, yPredTrain))})
     

    

    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # The mean squared error

    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % r2_score(yTest, yPred))

    # Plot outputs
    #print(yTest[:5,1])
    #print(xTest[:5,1])
    #plt.scatter(array_test[:,1], yTest,  color='black')
    #plt.plot(array_test[:,1], yPred, color='blue', linewidth=3)

    #uncomment to hide graph value information

trainingData = 0.2
targetIndex = 2
crossVal = False
split = True
featureComparison = False
#names=['Month', 'Hour', 'SurfTemp(TIRS)', 'PrecType', 'PrecAmount', 'SurfTemp(DST111)', 'Friction', 'SurfStatus'],
skipFeatures = [2]#, 4]#, 3]#, 1]#, 0]#, 6]#, 7]

#modelSurfaceTemperature([2,3,4,5,7,8], 0.2)
modelSurfaceTemperature(skipFeatures, trainingData, targetIndex, crossVal, split, featureComparison)