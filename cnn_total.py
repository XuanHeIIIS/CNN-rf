#!/usr/bin/python

import numpy as np
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from encode_data import data_process
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping

np.random.seed(1337)  # for reproducibility
batch_size = 32
nb_classes = 2
nb_epoch = 25
nb_filters = 320
seql = 81

def load_data(ppath, npath):
    print 'loading data......'
    data = data_process(ppath, npath)

    X = np.array([data[i][0] for i in xrange(len(data))])
    y = np.array([data[i][1] for i in xrange(len(data))])
    
    #Xn, X_test, yn, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    #X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(Xn, yn, test_size=0.1, random_state=0)
    
    # Get final model!!!
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)
    
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_cv = np_utils.to_categorical(y_cv, nb_classes)
    
    print 'data loaded success!'
    #return X_train, Y_train, X_test, Y_test, X_cv, Y_cv
    return X_train, Y_train, X_cv, Y_cv

def create_model():
    
    model = Sequential()

    #First convolutional layer
    model.add(Convolution2D(nb_filters, 4, 6, border_mode='valid', input_shape=(1, 4, seql), init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())

    #First pooling layer
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))

    #Second convolutional layer
    model.add(Convolution2D(nb_filters, 1, 6, border_mode='valid', input_shape=(1, 4, seql), init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())

    #Second pooling layer
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.5))

    #flatten
    model.add(Flatten())

    #full connected layer
    model.add(Dense(32, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    #full connected layer
    model.add(Dense(32, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    #output layer
    model.add(Dense(nb_classes, init='he_normal'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, momentum=0.92, decay=1e-6, nesterov=False)
    #adagrad = Adagrad(lr=0.1, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def training(X_train, Y_train, X_cv, Y_cv, model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, mode='min')
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_cv, Y_cv), callbacks=[early_stopping])
    print 'Model trained!'
    return model

def testing(X_test, Y_test, model):
    score = model.evaluate(X_test, Y_test, verbose=0)
    print 'Testing loss:', score[0]
    print 'Testing acc:', score[1]
    y_pred = model.predict(X_test)
    '''
    opath = 'pred_results'
    fout = open(opath, 'a')
    for i in xrange(y_pred.shape[0]):
        fout.write(str(Y_test[i]))
        fout.write('\t')
        fout.write(str(y_pred[i]))
        fout.write('\n')
    '''
    print 'Testing auc:', roc_auc_score(Y_test, y_pred)

def save_model(model):
    json_string = model.to_json()
    open('cnn_v3.json', 'w').write(json_string)
    model.save_weights('cnn_v3_weights.h5')

if __name__ == '__main__':
    ppath, npath = '../data/GSE63655/seqs/total_p81_seqs', '../data/GSE63655/seqs/total_n81_seqs'
    #X_train, Y_train, X_test, Y_test, X_cv, Y_cv = load_data(ppath, npath)
    X_train, Y_train, X_cv, Y_cv = load_data(ppath, npath)
    cnn_model = create_model()
    trained_model = training(X_train, Y_train, X_cv, Y_cv, cnn_model)
    #testing(X_test, Y_test, trained_model)
    save_model(trained_model)
