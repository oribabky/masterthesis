import warnings
#warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings('ignore')
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, make_scorer, confusion_matrix, accuracy_score, cohen_kappa_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from imblearn import over_sampling, under_sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import customlib #own library

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
#xl = pd.ExcelFile("../data/stations_e6/mätvärden 1520 2015-2016.xlsx")
#file = 'mätvärden 1520 2015-2016.xlsx'
file = 'mätvärden total.xlsm'
sheet = 'raw_noerr_featureengi'
#sheet = 'raw_noerr_undersampling'
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
print(dataset.groupby('PrecType').size())

#print(dataset.values)
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()


# histograms
#dataset.hist()
#plt.show()

# scatter plot matrix
#scatter_matrix(dataset[2:])
#plt.show()




def modelPrecipationType(skipFeatures, targetIndex, kFold, stratifiedFold, split):
    # Split-out validation dataset
    array = dataset.values
    X = customlib.sliceSkip2d(array, [0,1,2,3,4])
    Y = array[:, targetIndex]
    SEED = 7    
    testSize = 0
    if split:
        testSize = 0.2
    
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=testSize, random_state=SEED)

    #sm = SMOTE(random_state=SEED, ratio = 1.0)
    # Test options and evaluation metric

    #scoring = {'accuracy' : make_scorer(accuracy_score), 
           #'precision' : make_scorer(precision_score),
           #'recall' : make_scorer(recall_score), 
           #'f1_score' : make_scorer(f1_score)}

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('BP', MLPClassifier()))
    #models.append(('SVM', SVC())) # long run time...
    models.append(('RF', RandomForestClassifier()))

    # evaluate each model in turn
    results = []
    names = []
    #score = {'recall': make_scorer(recall_score(average='macro'))}
    if kFold:
        print("kfold:")
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=SEED)

            print(name)
            for score in ["accuracy", "f1_macro"]:
                print (score + " : " + 
                    str(model_selection.cross_val_score(model, xTrain, yTrain, cv=kfold, scoring=score).mean()))
        
            
    # evaluate each model in turn
    if stratifiedFold:
        print("stratified:")
        for name, model in models:
            stratified = model_selection.KFold(n_splits=10, random_state=SEED)

            print(name)
            for score in ["accuracy", "f1_macro"]:
                print (score + " : " + 
                    str(model_selection.cross_val_score(model, xTrain, yTrain, cv=stratified, scoring=score).mean()))
        

    overSampling = False
    underSampling = False

    if split:
        sm = SMOTE(random_state=SEED)
        #sm = RandomOverSampler(random_state=SEED)
        #sm = RandomUnderSampler(random_state=SEED)
        xTrainOS, yTrainOS = sm.fit_sample(xTrain, yTrain)

        for name, model in models:
            #cart = DecisionTreeClassifier()
            print("USING ORIGINAL DATASET")
            print(name)
            model.fit(xTrain, yTrain)
            predictions = model.predict(xTest)
            print(accuracy_score(yTest, predictions))
            print(cohen_kappa_score(yTest, predictions))
            print(confusion_matrix(yTest, predictions))
            print(classification_report(yTest, predictions))

            if overSampling:
                print("\nUSING OVERSAMPLED DATASET")
                print(name)
                model.fit(xTrainOS, yTrainOS)
                predictions = model.predict(xTest)
                print(accuracy_score(yTest, predictions))
                print(cohen_kappa_score(yTest, predictions))
                print(confusion_matrix(yTest, predictions))
                print(classification_report(yTest, predictions))

            if underSampling:
                print("\nUSING UNDERSAMPLED DATASET")
                print(name)
                model.fit(xTrainOS, yTrainOS)
                predictions = model.predict(xTest)
                print(accuracy_score(yTest, predictions))
                print(cohen_kappa_score(yTest, predictions))
                print(confusion_matrix(yTest, predictions))
                print(classification_report(yTest, predictions))


targetIndex = 3
kFold = False
stratifiedFold = False
split = True

#names=['Month', 'Hour', 'SurfTemp(TIRS)', 'PrecType', 'PrecAmount', 'SurfTemp(DST111)', 'Friction', 'SurfStatus'],
skipFeatures = [3]

modelPrecipationType(skipFeatures, 3, kFold, stratifiedFold, split)