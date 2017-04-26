# Simple library for routine operations

from sklearn import linear_model, svm, tree
import sklearn.metrics
import numpy as np

SVM_LINEAR_MODEL = 'SVM_LINEAR'
EVALUATION_TYPE = 'ACCURACY'
FILE = '../data/1CSurr.csv'
TRAIN_RATIO = 0.25
import random
import copy

# TODO: make fns tostore pickle objects of the model

# reads from file and returns X, Y
def ReadFromFile(filename,shuffle=None):
    with open(filename) as f:
        data = np.loadtxt(f, delimiter=",")
    if shuffle is not None:
        np.random.shuffle(data)
    X = np.array((data[:, 0:-1]))
    X.tolist()
    Y = np.array(data[:, -1])
    return X, Y


def SplitTrainAndTest(train_ratio, X, Y):
    train_size = int(len(Y) * train_ratio)
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    return [X_train, Y_train, X_test, Y_test]


def TrainModel(X, Y, model_type='SVM_LINEAR'):
    if model_type == SVM_LINEAR_MODEL:
        model = svm.SVC(kernel='linear')
    elif model_type =='SVM_LINEAR_HIGHC':
        model = svm.SVC(kernel='linear',C=10, class_weight='auto')
    elif model_type=='SVM_ONECLASS':
        model=svm.OneClassSVM(nu=0.01,kernel="rbf",gamma=0.1)
        model.fit(X)
        return model
    elif model_type=="DT":
        model = tree.DecisionTreeClassifier()#class_weight=None)#'auto')
    elif model_type=='Testing_SVM':
        model=svm.SVC(kernel='linear', C=1)
    else:
        model=linear_model.LogisticRegression(class_weight='auto')

    if model_type == 'SVM_X':
        X_copy=copy.deepcopy(X)
        for i in X_copy:
            i[1]=0
        model = svm.SVC(kernel='linear')
        model.fit(X_copy,Y)
    else:
        model.fit(X, Y)
    return model


def ComputePerf(Y_actual, Y_pred):
    conf_matrix = sklearn.metrics.confusion_matrix(Y_actual, Y_pred)
    if EVALUATION_TYPE == 'ACCURACY':
        metric = sklearn.metrics.accuracy_score(Y_actual, Y_pred)
    else:
        metric = sklearn.metrics.f1_score(Y_actual, Y_pred)
    return {'metric': metric, 'conf_matrix': conf_matrix}


def PredictModel(model, X_test, Y_test=None):

    Y_pred = model.predict(X_test)
    return Y_pred

def TestPipeline():
    X, Y = ReadFromFile(FILE)
    [X_train, Y_train, X_test, Y_test] = SplitTrainAndTest(TRAIN_RATIO, X, Y)
    model = TrainModel(X_train, Y_train)

    performance = ComputePerf(PredictModel(model, X_train), Y_train)
    print 'Train Performance on %s: %s' % (EVALUATION_TYPE,
                                           performance['metric'])

    performance = ComputePerf(PredictModel(model, X_test), Y_test)
    print 'Test Performance on %s: %s' % (EVALUATION_TYPE,
                                          performance['metric'])

def WriteToFile(filename,X,Y=None):
    Y=[int(y) for y in Y]
    X=np.concatenate((X,np.array([list(Y)]).T),axis=1)

    with open(filename,'w') as f:
        data = np.savetxt(f,X,delimiter=",")

if __name__ == '__main__':
    TestPipeline()
