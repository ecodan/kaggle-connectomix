__author__ = 'dan'

'''
Variation of Step 3 of pipeline for NN (uses graph2 format)

Use the following methods as needed.

def evaluate: utility method for experimenting with and tuning classifiers
def train: fit the classifier and scalar
def predict: run the data sets through the trained classifier

'''

import networkx as nx
from datetime import datetime
import numpy as np
import pandas as pd
import itertools
import math
import sklearn as sl
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import sklearn.preprocessing as pre
import pickle
import os

import pybrain as pb
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import time
from pybrain.structure import TanhLayer
from pybrain.structure import SigmoidLayer

# prep_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt'])]
# prep_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC02inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC02inh.txt','networkPositions_iNet1_Size100_CC02inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC03inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC03inh.txt','networkPositions_iNet1_Size100_CC03inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC04inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC04inh.txt','networkPositions_iNet1_Size100_CC04inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC05inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC05inh.txt','networkPositions_iNet1_Size100_CC05inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC06inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC06inh.txt','networkPositions_iNet1_Size100_CC06inh.txt'])]
# test_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt'])]
# train_nets = prep_nets

prep_nets = [
#     ('normal-1',['fluorescence_normal-1.txt.desc.csv-graph2.graphml','network_normal-1.txt','networkPositions_normal-1.txt'])
    ('normal-2',['fluorescence_normal-2.txt.desc.csv-graph2.graphml','network_normal-2.txt','networkPositions_normal-2.txt'])
    ('normal-3',['fluorescence_normal-3.txt.desc.csv-graph2.graphml','network_normal-3.txt','networkPositions_normal-3.txt'])
    ('normal-3-highrate',['fluorescence_normal-3-highrate.txt.desc.csv-graph2.graphml','network_normal-3-highrate.txt','networkPositions_normal-3-highrate.txt'])
    ('normal-4',['fluorescence_normal-4.txt.desc.csv-graph2.graphml','network_normal-4.txt','networkPositions_normal-4.txt'])
    ('normal-4-lownoise',['fluorescence_normal-4-lownoise.txt.desc.csv-graph2.graphml','network_normal-4-lownoise.txt','networkPositions_normal-4-lownoise.txt'])
    ('highcc',['fluorescence_highcc.txt.desc.csv-graph2.graphml','network_highcc.txt','networkPositions_highcc.txt'])
    ('highcon',['fluorescence_highcon.txt.desc.csv-graph2.graphml','network_highcon.txt','networkPositions_highcon.txt'])
    ('lowcc',['fluorescence_lowcc.txt.desc.csv-graph2.graphml','network_lowcc.txt','networkPositions_lowcc.txt'])
    ('lowcon',['fluorescence_lowcon.txt.desc.csv-graph2.graphml','network_lowcon.txt','networkPositions_lowcon.txt'])
#     ('valid', ['fluorescence_valid.txt.desc.csv-graph2.graphml','','networkPositions_valid.txt']),
#     ('test', ['fluorescence_test.txt.desc.csv-graph2.graphml','','networkPositions_test.txt'])
]

train_nets = [
    ('normal-1',['fluorescence_normal-1.txt.desc.csv-graph2.graphml','network_normal-1.txt','networkPositions_normal-1.txt'])
]
# test_nets = [
#     ('valid',['fluorescence_valid.txt.desc.csv-graph2.graphml','','networkPositions_valid.txt']),
#     ('test',['fluorescence_test.txt.desc.csv-graph2.graphml','','networkPositions_test.txt'])
# ]


model_file = '/Users/dan/dev/datasci/kaggle/connectomix/out/model2_nn.pkl'

in_dir = '/Users/dan/dev/datasci/kaggle/connectomix/'

_cols = [
    # ['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14','e15','d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15'],
    # ['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15'],
    # ['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14','e15'],
    ['d1','d2','d4','d8']
]

def time_remaining(tot_iters, cur_iter, total_dur):
    avg = total_dur/((cur_iter if cur_iter != 0 else 1)*1.0)
    rem = (tot_iters - cur_iter) * avg
    return avg/1000, rem/1000


def evaluate(in_dir, nets):
    print('evaluate...')

    best_auc = 0.0

    for t in nets:
        k = t[0]
        v = t[1]
        print(k + ': evaluating...')
        out_dir = in_dir + k + '/out/'

        # read tables
        dfc = pd.read_table(out_dir + v[0] + '.test.csv', sep=',', index_col=[0,1])

        # read actuals
        actuals = pd.read_table(in_dir + k + '/' + v[1], sep=',', dtype={0:np.int32, 1:np.int32, 2:np.int32}, header=None)
        actuals.columns = [0,1,'act']
        actuals = actuals.groupby([0,1]).sum()

        # merge with actuals
        dfj = dfc.join(actuals)
        dfj['act'] = dfj['act'].apply(lambda x: 1 if x == 1 else 0)
        dfj.to_csv(out_dir + v[0] + '.test.act.csv')

        # get rid of identity values
        print('DIAG dfj=' + str(dfj.shape))
        dft = dfj[dfj['i'] != dfj['j']]
        # dft = dfj
        print('DIAG dft=' + str(dft.shape))

        for cols in _cols:
            print('starting for cols=' + str(cols))

            print('quick classifier...')
            # quick classifier
            X = dft[cols]
            y = dft['act']

            scaler = pre.StandardScaler()
            # scaled = scaler.fit_transform(X)
            # X = pd.DataFrame(scaled, columns=X.columns)

            # this is for a baseline
            clf = LogisticRegression(C=1,penalty='l1', class_weight={0:1,1:8})
            # clf = RandomForestClassifier(n_estimators=10, max_features=.5)
            scores = sl.cross_validation.cross_val_score(clf, X, y, scoring='roc_auc')
            mscores = np.mean(scores)
            print(k + ': xval scores=' + str(scores) + ' | mean=' + str(mscores))
            if mscores > best_auc:
                best_auc = mscores

            # neural net
            print('NN')
            X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.2, test_size=0.1)
            # for h_size in [2,4,8,12]:
            #     for _bias in [True, False]:
            #         for h_class in [TanhLayer, SigmoidLayer]:
            # score to beat: 2/True/Tan = 0.747097371135

            for h_size in [3]:
                for _bias in [True]:
                    for h_class in [TanhLayer]:
                        print(k + ': NN iter ' + str(h_size) + ' ' + str(_bias) + ' ' + str(h_class))
                        net = buildNetwork(X_train.shape[1], X_train.shape[1] * h_size, X_train.shape[1] * h_size, 1, bias = _bias, hiddenclass=h_class)
                        ds = SupervisedDataSet(X_train.shape[1], 1)
                        ds.setField('input', X_train)
                        ds.setField('target', y_train.reshape(-1, 1))
                        trainer = BackpropTrainer(net, ds)
                        trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 100, continueEpochs = 10 )

                        ds = SupervisedDataSet(X_test.shape[1], 1)
                        ds.setField('input', X_test)
                        ds.setField('target', y_test.reshape(-1,1))
                        z = net.activateOnDataset(ds)
                        # print (k + ' NN results:\n' + str(list(z)))
                        fpr, tpr, t = sl.metrics.roc_curve(y_test, z )
                        nn_auc = sl.metrics.auc(fpr, tpr)
                        print (k + ': NN roc auc: ' + str(nn_auc))
                        if nn_auc > best_auc:
                            best_auc = nn_auc

    print ('done; best_auc=' + str(best_auc))

def train(in_dir, nets):
    print('train...')

    if len(nets) > 1:
        print('ERROR can only support one training net since only one model file supported')
        return

    for t in nets:
        k = t[0]
        v = t[1]
        print(k + ': training at ' + str(datetime.now()))

        out_dir = in_dir + k + '/out/'

        # read tables
        print(k + ': reading ' + out_dir + v[0] + '.test.csv')
        dfc = pd.read_table(out_dir + v[0] + '.test.csv', sep=',', index_col=[0,1])

        # read actuals
        print(k + ': reading ' + in_dir + k + '/' + v[1])
        actuals = pd.read_table(in_dir + k + '/' + v[1], sep=',', dtype={0:np.int32, 1:np.int32, 2:np.int32}, header=None)
        actuals.columns = [0,1,'act']
        actuals = actuals.groupby([0,1]).sum()

        # merge with actuals
        dfj = dfc.join(actuals)
        dfj['act'] = dfj['act'].apply(lambda x: 1 if x == 1 else 0)

        scaler = pre.StandardScaler()

        # quick classifier
        dft = dfj[dfj['i'] != dfj['j']]

        cols = _cols
        X = dft[cols]
        y = dft['act']

        X_train,X_test,y_train,y_test = train_test_split(X, y)

        print(k + ': fitting model')
        h_size = 2
        _bias = True
        h_class = TanhLayer
        _max_epochs = 1000

        net = buildNetwork(X_train.shape[1], X_train.shape[1] * h_size, X_train.shape[1] * h_size, 1, bias = _bias, hiddenclass=h_class)

        ds = SupervisedDataSet(X.shape[1], 1)
        ds.setField('input', X)
        ds.setField('target', y.reshape(-1, 1))

        trainer = BackpropTrainer(net, ds)
        trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = _max_epochs, continueEpochs = 10 )

        s = pickle.dumps(net)
        f = open(model_file, 'w')
        f.write(s)
        f.close()

        print('testing model...')
        f = open(model_file, 'r')
        s2 = f.read()
        net2 = pickle.loads(s2)
        ds = SupervisedDataSet(X_test.shape[1], 1)
        ds.setField('input', X_test)
        ds.setField('target', y_test.reshape(-1,1))
        z = net2.activateOnDataset(ds)

        # this is probably a crappy way to do this
        fpr, tpr, t = sl.metrics.roc_curve(y_test, z )
        nn_auc = sl.metrics.auc(fpr, tpr)
        print (k + ': NN roc auc: ' + str(nn_auc))

def predict(in_dir, model_file, nets):
    print('predict...')

    res = np.array([])

    for t in nets:
        k = t[0]
        v = t[1]
        print(k + ': predicting at ' + str(datetime.now()))
        out_dir = in_dir + k + '/out/'

        # read tables
        dft = pd.read_table(out_dir + '/' + v[0] + '.test.csv', sep=',', index_col=[0,1])

        cols = _cols
        X = dft[cols]
        y = np.zeros(len(X))

        ds = SupervisedDataSet(X.shape[1], 1)
        ds.setField('input', X)
        ds.setField('target', y.reshape(-1,1))

        f = open(model_file, 'r')
        s2 = f.read()
        net2 = pickle.loads(s2)
        preds = net2.activateOnDataset(ds)
        probs = preds

        X['key'] = dft[['i','j']].apply(lambda x: k + '_' + str((int)(x['i'])) + '_' + str((int)(x['j'])), axis=1)
        X['pred'] = probs
        if len(res) == 0:
            res = X[['key', 'pred']].values
        else:
            res = np.concatenate([res, X[['key', 'pred']].values])
        print(k + ': done predicting; k size=' + str(len(X))+ ' | res size=' + str(len(res)))

    print('writing final output')
    df = pd.DataFrame(res,columns=['NET_neuronI_neuronJ','Strength'])
    df.to_csv(in_dir + '/out/predictions.csv', index=False)
    print('done; num rows=' + str(len(df)))

prepare(in_dir, prep_nets)

# evaluate(in_dir, train_nets)

# scaler = train(in_dir, train_nets)

# predict(in_dir, model_file, test_nets, scaler)


# still to try:
# - neural net
# -- increase hidden layers
# -- increase epochs
# x KNN - default config got .732
# x adjusting features
# - different graph model treatment of burst periods
# - ensemble between correlation and graph approach
# x create 8x8 pre-active probability matrix on each connection
# - remove directionality
