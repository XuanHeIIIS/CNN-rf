#!/usr/bin/python

import numpy as np
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import MinMaxScaler
from utils import seq2mat
from theano import function
from keras import backend as K
import cPickle
import sys
import getopt

def usage():
    print 'Usage:%s [-h|-s|-i|-o] [--help|--species|--input|--output] args....' % sys.argv[0]
    print 'the species are allowed in \'human\' and \'mus\'...'

def get_opt():
    sp = ''
    out = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hs:i:o:', ['help', 'species=', 'input=', 'output='])
    except getopt.GetoptError:
        print 'Please provide right parameters!'
        usage()
        sys.exit(1)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(1)
        elif opt in ('-s', '--species'):
            sp = arg
        elif opt in ('-i', '--input'):
            ipath = arg
        elif opt in ('-o', '--output'):
            opath = arg
        else:
            usage()
            sys.exit(1)
    return sp, ipath, opath

def load_seqs(ipath):
    print 'loading sequences...'
    genes = []
    fin = open(ipath, 'r')
    while 1:
        line = fin.readline().strip()
        if len(line) == 0:
            break
        elif '>' in line:
            name = line[1:]
            seq = fin.readline().strip()
        else:
            continue
        genes.append((name, seq))
    fin.close()
    print 'Sequences were loaded!'
    return genes

def coding(gene):
    print 'Sequences are coding into numeric data......'
    flank = 40
    data = []
    name = gene[0]
    seq = gene[1]
    l = len(seq)
    seq = 'N'*flank+seq+'N'*flank
    for i in xrange(l):
        mp = flank + i
        subseq = ''
        if seq[mp] == 'T':
            subseq = seq[mp-flank:mp+flank+1]
            mat = seq2mat([subseq])
            data.append([mat, i, name])
        else:
            pass

    X = np.array([data[i][0] for i in xrange(len(data))])
    idxs = [data[i][1] for i in xrange(len(data))]
    names = [data[i][2] for i in xrange(len(data))]

    print 'Sequence coded!'
    return X, idxs, names

def load_cnn(sp):
    print 'loading trained cnn model...'
    model = None
    if sp == 'mus':
        model = model_from_json(open('cnn_mus.json').read())
        model.load_weights('cnn_mus_weights.h5')
    elif sp == 'human':
        model = model_from_json(open('cnn_human.json').read())
        model.load_weights('cnn_human_weights.h5')
    else:
        print 'ERROR! Please check your species parameter!'
        usage()
        exit(1)
    print 'cnn model loaded!'
    return model

def get_layer(model, X):
    get_feature = K.function([model.layers[0].input, K.learning_phase()], [model.layers[12].output])
    feature = get_feature([X, 0])[0]
    print feature.shape
    scaler = MinMaxScaler()
    nX = scaler.fit_transform(feature)
    return nX

def load_rf(sp):
    print 'loading trained random forest model...'
    rfm = None
    if sp == 'mus':
        rfm = cPickle.load(open('mus_rf.pkl','rb'))
    elif sp == 'human':
        rfm = cPickle.load(open('human_rf.pkl','rb'))
    else:
        print 'ERROR! Please check your species parameter!'
        usage()
        exit(1)
    print 'random forest model loaded!'
    return rfm

def prediction(rfm, nX, idxs, names, opath):
    fout = open(opath, 'a')
    preds = rfm.predict_proba(nX)[:, 1]
    for i in xrange(len(idxs)):
        op = '\t'.join([names[i], str(idxs[i]), str(preds[i])])
        fout.write(op)
        fout.write('\n')
    print 'Current gene predicted!'
    fout.close()

if __name__ == '__main__':
    #path = '/home/hx/Research/genome_sequence/human/ucsc_RefSeq'
    #path = '../data/scarlet_RNA_seqs'
    sp, ipath, opath = get_opt()
    model = load_cnn(sp)
    rfm = load_rf(sp)
    genes = load_seqs(ipath)
    for gene in genes:
        X, idxs, names = coding(gene)
        nX = get_layer(model, X)
        prediction(rfm, nX, idxs, names, opath)
    print 'Job Done!'
