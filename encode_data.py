#!/usr/bin/python

import numpy as np
from utils import seq2mat
np.random.seed(1337)

def data_process(ppath, npath):
    pfin = open(ppath, 'r')
    nfin = open(npath, 'r')
    data = []
    for pseq in pfin:
        pseq = pseq.strip()
        if len(pseq) == 0:
            continue
        mat1 = seq2mat([pseq])
        data.append([mat1, 1])
    for nseq in nfin:
        nseq = nseq.strip()
        if len(nseq) == 0:
            continue
        mat2 = seq2mat([nseq])
        data.append([mat2, 0])
    np.random.shuffle(data)
    np.random.shuffle(data)
    np.random.shuffle(data)
    pfin.close()
    nfin.close()
    return data

