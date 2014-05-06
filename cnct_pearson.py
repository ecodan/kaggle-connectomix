__author__ = 'dan'

import networkx as nx
from datetime import datetime
import numpy as np
import pandas as pd
import itertools
import scipy.stats as stats
import os

nets = [('small','fluorescence_iNet1_Size100_CC01inh.txt.diff.csv')]
out_dir = '/Users/dan/dev/datasci/kaggle/connectomix/out/'
#nets = {'valid':'fluorescence_valid.txt.desc.csv', 'test':'fluorescence_test.txt.desc.csv'}
#out_dir = '/Users/dan/dev/datasci/kaggle/connectomix/out/'
in_dir = '/Users/dan/dev/datasci/kaggle/connectomix/'

start = datetime.now()
last = datetime.now()

def pearson_analyze(in_dir, out_dir, nets):
    print('starting pearson analyze')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    res = []
    for t in nets:
        k = t[0]
        v = t[1]
        print('reading discretized file ' + in_dir + k + '/out/' + v )
        n_act = pd.read_table(in_dir + k + '/out/' + v, sep=',', header=None)

        # set the columns headers to 1-based series
        neuron_ct = len(n_act.columns)
        n_act.columns = range(1, neuron_ct+1)

        # loop through all neuron combinations and calculate pearson coeff
        for i in range(1,neuron_ct+1):
            if i % 10 == 0:
                print('on neuron ' + str(i))
            for j in range(1,neuron_ct+1):
                key = k + '_' + str(i) + '_' + str(j)
                if i == j:
                    res.append([key, 0.0])
                else:
                    corr = str(stats.pearsonr(n_act[i], n_act[j])[0])
                    res.append([key, corr])

    df = pd.DataFrame(res,columns=['NET_neuronI_neuronJ','Strength'])
    df.to_csv(out_dir + '/predictions-pearson.csv', index=False)


pearson_analyze(in_dir, out_dir, nets)