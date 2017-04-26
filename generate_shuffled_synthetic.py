
## Script to generate datasets for Detectability and False alarm experiments. 

import data_infra # Library file for routine operations
import numpy as np
import matplotlib.pyplot as plt
import pprint, math
from sklearn.feature_selection import SelectKBest
import sklearn.ensemble as ensemble_lib
from sklearn import linear_model, svm, tree

# Parameters to change 
file='fin_wine.csv'
FILENAME='induced_datasets/datasets/shuffled_'+file
class_change=1 # Need to specify class 1 or 0, for some datasets 1 leads to a realiable drop, need to evaluate both and pick the one which suits the data
change_point=0.5 # Ratio of stream after which drift needs to occur
N_shuffle=0.25 # Ratio of features, the drift needs to affect
first=True # first= True means the most important features will be affected, =False if you want to generate a false alarm experiment.
PRINT_DETAILS=True # prints scirpt output


## Read file and shuffle to remove stray concept drift

[X,Y]=data_infra.ReadFromFile(FILENAME, shuffle = True)
[X_train, Y_train, X_test, Y_test]=data_infra.SplitTrainAndTest(change_point, X, Y)
N_dim=len(X[0])
N_shuffle=int(N_shuffle*N_dim)


## get feature importance list
selection=SelectKBest(k=len(X[0]))
model=selection.fit(X,Y)
x=[i for i in range(len(X[0]))]
scores=model.scores_
scores=[i if not math.isnan(i) else 0 for i in scores]
yx=zip(scores,x)
yx.sort(reverse=True)
# Uncomment to check ranking - print yx

## Checking previous model
model = data_infra.TrainModel(X,Y)
# You can also use random subspace model here
previous_accuracy=data_infra.ComputePerf(Y_test, model.predict(X_test))['metric']
print 'Train accuracy ', data_infra.ComputePerf(Y_train,model.predict(X_train))['metric']
print 'Test accuracy ', data_infra.ComputePerf(Y_test, model.predict(X_test))['metric']

## Shuffle features based on importance factor
shuffle_list=range(N_dim)
# Round robin split
if first:
        prev=shuffle_list[yx[0][1]]
        for i in range(N_shuffle):
            shuffle_list[yx[i][1]]=shuffle_list[yx[(i+1)%N_shuffle][1]]
        shuffle_list[yx[i][1]]=prev
else:
        prev=shuffle_list[yx[N_dim-N_shuffle][1]]
        for i in range(N_dim-N_shuffle, N_dim):
            shuffle_list[yx[i][1]]=shuffle_list[yx[(i+1)%N_dim][1]]
        shuffle_list[yx[i][1]]=prev

for i in range(len(X_test)):
    if Y_test[i]==class_change:
        X_test[i,:]=X_test[i,shuffle_list]

if PRINT_DETAILS:
    print '*************'
    print ' # Instances, # attributes: ', len(Y), len(X[0])
    print 'change class, N_shuffle, first: ', class_change, N_shuffle, first
    print 'change point: ', len(X_train)
    new_accuracy=data_infra.ComputePerf(Y_test, model.predict(X_test))['metric']
    print 'Test accuracy: ',data_infra.ComputePerf(Y_test, model.predict(X_test))['metric']
    print 'Drop in accuracy after drift:',  previous_accuracy-new_accuracy
    print '*************'

# Output is written to file

if first:
      FILENAME='induced_datasets/shuffled_attributes/first_'+file
else:
      FILENAME='induced_datasets/shuffled_attributes/last_'+file
X=np.concatenate((X_train, X_test))
data_infra.WriteToFile(FILENAME, X, Y)


