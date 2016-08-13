#!/usr/bin/python

import numpy as np
from keras.utils import np_utils
from keras import models
from keras.models import model_from_json
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import cPickle as cP
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import MinMaxScaler
from theano import function
from keras import backend as K
import matplotlib.pyplot as plt

def load_data(dpath, fold):
    print 'loading data......'
    fin = open(dpath, 'rb')
    datas = cP.load(fin)
    print 'Folds:', len(datas)

    tag = 0
    X_test = np.array([])
    y_test = np.array([])
    folds = range(len(datas))
    if fold in folds:
        tag = 1
        del folds[fold]
        X_test = np.array([datas[fold][i][0] for i in xrange(len(datas[0]))])
        y_test = np.array([datas[fold][i][1] for i in xrange(len(datas[0]))])
        print 'X_test shape:', X_test.shape
    
    X_train = []
    y_train = []
    for n in folds:
        X_train.extend([datas[n][i][0] for i in xrange(len(datas[n]))])
        y_train.extend([datas[n][i][1] for i in xrange(len(datas[n]))])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print 'X_train shape:', X_train.shape
    
    print 'data loaded success!'
    if tag == 1:
        return X_test, y_test, X_train, y_train
    else:
        return X_train, y_train

def load_cnn():
    print 'loading trained cnn model...'
    model = model_from_json(open('cnn_human.json').read())
    model.load_weights('cnn_human_weights.h5')
    return model

def get_layer(model, X):
    get_feature = K.function([model.layers[0].input, K.learning_phase()], [model.layers[11].output])
    feature = get_feature([X, 0])[0]
    print feature.shape
    scaler = MinMaxScaler()
    nX = scaler.fit_transform(feature)
    return nX

def rf_model(X_tr, y_train, X_tt, y_test):
    rf = RFC(n_estimators=100, criterion='entropy', min_samples_split=1, n_jobs=12, random_state=0)
    rf.fit(X_tr, y_train)
    #cP.dump(rf, open("./human_rf.pkl","wb"))
    y_tt = np_utils.to_categorical(y_test, 2)
    y_pred = rf.predict_proba(X_tt)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    y_p = rf.predict(X_tt)
    print '*' * 40
    print 'Testing acc:'
    print accuracy_score(y_test, y_p)
    print 'Testing auc score:'
    print roc_auc_score(y_tt, y_pred)
    print '*' * 40
    #plt.figure()
    #plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % auc(fpr, tpr))
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.0])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Predict Psi sites')
    #plt.legend(loc="lower right")
    #plt.savefig('./results/human_test.png')
    #plt.show()
    return rf

if __name__ == '__main__':
    dpath = '../data/rmbase/h101_data.pkl'
    X_test, y_test, X_train, y_train = load_data(dpath, 9)
    model = load_cnn()
    X_tr = get_layer(model, X_train)
    X_tt = get_layer(model, X_test)
    rf = rf_model(X_tr, y_train, X_tt, y_test)

    ########### Test on mus data ############

    tdpath = '../data/rmbase/m101_data.pkl'
    tX, ty = load_data(tdpath, 10)
    tnX = get_layer(model, tX)
    tty = np_utils.to_categorical(ty, 2)
    ty_pred = rf.predict_proba(tnX)
    tfpr, ttpr, thresholds = roc_curve(ty, ty_pred[:, 1])
    ty_p = rf.predict(tnX)
    print '*' * 40
    print 'Testing acc:'
    print accuracy_score(ty, ty_p)
    print 'Testing auc score of mus data:'
    print roc_auc_score(tty, ty_pred)
    #plt.figure()
    #plt.plot(tfpr, ttpr, label='ROC curve (area = %0.4f)' % auc(tfpr, ttpr))
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.0])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Prediction of Psi sites')
    #plt.legend(loc="lower right")
    #plt.savefig('./results/human_to_mus_test.png')
    #plt.show()
