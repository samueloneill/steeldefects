from datetime import datetime
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as mpl
import pyautogui 
from typing import Counter
import hickle as hkl
#SciKit learn libraries
from sklearn import neighbors, svm, tree, naive_bayes, ensemble
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler

#Import datasets (CSV files must be in same directory as this script)
trainData = pd.read_csv('train.csv') #Training Dataset 
testData = pd.read_csv('test.csv') #Testing Dataset

#Summarise the data using statistical measures
trainData.describe()
testData.describe()
categories = trainData['TypeOfDefects'].nunique()
defectType = ["Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other"]
totalSamples = 0
print("Number of samples in each category:")
print(trainData['TypeOfDefects'].value_counts().sort_index())

#Split training data into attribute only/label only sets
columns = len(trainData.columns)
attributes = columns-1
trainAtts_old = trainData.iloc[:,0:attributes]
trainLabels_old = trainData.iloc[:,attributes]

#Oversample dataset to handle imbalanced samples
over = RandomOverSampler(random_state=42)
trainAtts, trainLabels = over.fit_resample(trainAtts_old, trainLabels_old)
print("Balanced by oversampling: ",  str(Counter(trainLabels))[9:-2])
print("\nAttribute dataset shape: {}\nLabel dataset shape: {}".format(shape(trainAtts),shape(trainLabels)))

#Data visualisation using various plots
#Configure image size
screenWidth, screenHeight = pyautogui.size()
imageSize = mpl.rcParams["figure.figsize"]
imageSize[0] = screenWidth
imageSize[1] = screenHeight
if pyautogui.confirm("Show data plots?")=='OK': #Avoid unecessary loading of plots by asking for user input
    trainAtts.hist(layout=(4,7))  #Histogram
    trainAtts.plot(kind='box', subplots=True, layout=(4,7)) #Box plot
    mpl.show()

#In order to find accuracy of the model, we need a training and testing dataset for comparisons
#The testing dataset is 10% of the training dataset
#It is NOT 'test.csv' - this has no labels and so is used for submission, not accuracy testing
#'X' variables are attributes, 'y' variables are labels
Xtrain_old, Xtest_old, ytrain_old, ytest_old = train_test_split(trainAtts, trainLabels, test_size=0.1)

#Convert datasets to numpy objects for model building
Xtrain = Xtrain_old.to_numpy()
Xtest = Xtest_old.to_numpy()
ytrain = ytrain_old.to_numpy()
ytest = ytest_old.to_numpy()
#Print out the shape of each dataset to verify the split was correct
print("\nAfter splitting:")
print("Training: Attribute dataset = {}, Label dataset = {}".format(Xtrain.shape, ytrain.shape))
print("Testing: Attribute dataset = {}\n".format(Xtest.shape))

#Initial model training
modelList = []
modelNames = []
trainingResults = []
modelList.append(('KNN', neighbors.KNeighborsClassifier()))
modelList.append(('SVC', svm.SVC()))
modelList.append(('TREE', tree.DecisionTreeClassifier()))
modelList.append(('NB', naive_bayes.GaussianNB()))
modelList.append(('GBM', ensemble.GradientBoostingClassifier()))
modelList.append(('RF', ensemble.RandomForestClassifier()))
#For loop cycles through each model and calculates its accuracy
for name, model in modelList:
    startTime = datetime.now()
    kfold = KFold(n_splits=10, shuffle=True, random_state=888)
    crossVal = cross_val_score(model, Xtrain, ytrain, cv=kfold, scoring='accuracy')
    trainingResults.append(crossVal)
    modelNames.append(name)
    endTime = datetime.now() - startTime
    print("{}: Accuracy={}, Training time={}".format(name, crossVal.mean(), endTime))

#Selected random forest as model to be fine-tuned using random search
rf = ensemble.RandomForestClassifier()
print("\nRF Default hyperparameters:")
print(rf.get_params())
#Create random search parameter grid
n_estimators = [int(x) for x in np.linspace(start=200,stop=2000,num=10)] #number of trees
max_features = ['auto', 'sqrt'] #number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10,110,num=11)] #maximum levels in tree
max_depth.append(None)
min_samples_split = [2,5,10] #minimum samples required to split a node
min_samples_leaf = [1,2,4] #minimum samples at each leaf node
bootstrap = [True,False] #method of selecting samples to train each tree
randomGrid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#Begin random search tuning
rfRandSearch = RandomizedSearchCV(estimator=rf, param_distributions=randomGrid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rfRandSearch.fit(Xtrain, ytrain)
print("\nTuned RF hyperparameters:")
print(rfRandSearch.best_params_)

#Compare tuned model and base model
def checkAccuracy(model, Xtest, ytest):
    prediction = model.predict(Xtest)
    errors = abs(prediction - ytest)
    accuracy = 100 - (100*np.mean(errors/ytest))
    print("Avg error = {:0.4f}degrees\nAccuracy = {:0.2f}%".format(np.mean(errors),accuracy))
    return accuracy

baseModel = ensemble.RandomForestClassifier()
baseModel.fit(Xtrain, ytrain)
print("\nBase performance: ")
baseAccuracy = checkAccuracy(baseModel, Xtest, ytest)
tunedModel = rfRandSearch.best_estimator_
print("\nTuned performance: ")
tunedAccuracy = checkAccuracy(tunedModel, Xtest, ytest)
if tunedAccuracy > baseAccuracy:
    print("Tuning successful! Best accuracy: {} (improvement of {}".format(tunedAccuracy, (tunedAccuracy-baseAccuracy)))
else:
    print("Tuning returned no improvements. Best accuracy: {}".format(baseAccuracy))

#Retrain the tuned model using the entire original training dataset
finalModel = ensemble.RandomForestClassifier(n_estimators=800, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=90, bootstrap=False)
finalModel.fit(trainAtts_old, trainLabels_old)
#Save the final model
filename = "879282_finalModel.h5"
hkl.dump(finalModel, filename, mode='w')
print("Saved final model")

#Use saved model to predict classification results of test.csv
load_hkl = hkl.load(filename)
finalPredictions = load_hkl.predict(testData)
#Save predictions to CSV file
df = pd.DataFrame({'TypeOfDefects':finalPredictions})
df.index = df.index + 1  #Start CSV index from 1, not 0
df.to_csv('UP879282_Predictions.csv', index_label='IndexOfTestSample')
print("Saved predictions")