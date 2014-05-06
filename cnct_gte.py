__author__ = 'dan'

import networkx as nx
from datetime import datetime
import numpy as np
import pandas as pd
import itertools
import scipy.stats as stats
import os
import mlalgorithms.transfer_entropy

nets = [('small','fluorescence_iNet1_Size100_CC01inh.txt.desc.csv')]
out_dir = '/Users/dan/dev/datasci/kaggle/connectomix/out/'
#nets = {'valid':'fluorescence_valid.txt.desc.csv', 'test':'fluorescence_test.txt.desc.csv'}
#out_dir = '/Users/dan/dev/datasci/kaggle/connectomix/out/'
in_dir = '/Users/dan/dev/datasci/kaggle/connectomix/'

def gte_analyze(in_dir, out_dir, nets):
    print('starting gte analyze')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    res= []
    for t in nets:
        k = t[0]
        v = t[1]
        print('reading discretized file ' + in_dir + k + '/out/' + v )
        n_act = pd.read_table(in_dir + k + '/out/' + v, sep=',', header=None)

        # set the columns headers to 1-based series
        neuron_ct = len(n_act.columns)
        n_act.columns = range(1, neuron_ct+1)

        # call GTE
        entropy = mlalgorithms.transfer_entropy.GTE(n_act.values, 3)

        # reformat
        for i in range(0, neuron_ct):
            if i % 10 == 0:
               print('on neuron ' + str(i))
            for j in range(0, neuron_ct):
                key = k + '_' + str(i) + '_' + str(j)
                res.append([key, entropy[i][j]])

    df = pd.DataFrame(res,columns=['NET_neuronI_neuronJ','Strength'])
    df.to_csv(out_dir + '/predictions-gte.csv', index=False)


gte_analyze(in_dir, out_dir, nets)