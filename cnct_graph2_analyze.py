__author__ = 'dan'

'''
Step 3 of pipeline

Use the following methods as needed.

def prepare: converts the graphml file into a matrix for classification
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
import time
from sklearn.ensemble import GradientBoostingClassifier

# prep_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt'])]
# prep_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC02inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC02inh.txt','networkPositions_iNet1_Size100_CC02inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC03inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC03inh.txt','networkPositions_iNet1_Size100_CC03inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC04inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC04inh.txt','networkPositions_iNet1_Size100_CC04inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC05inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC05inh.txt','networkPositions_iNet1_Size100_CC05inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC06inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC06inh.txt','networkPositions_iNet1_Size100_CC06inh.txt'])]
# test_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph2.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt'])]
# train_nets = prep_nets

# prep_nets = [
    #     ('normal-1',['fluorescence_normal-1.txt.desc.csv-graph2.graphml','network_normal-1.txt','networkPositions_normal-1.txt']),
    # ('normal-2',['fluorescence_normal-2.txt.desc.csv-graph2.graphml','network_normal-2.txt','networkPositions_normal-2.txt']),
    # ('normal-3',['fluorescence_normal-3.txt.desc.csv-graph2.graphml','network_normal-3.txt','networkPositions_normal-3.txt']),
    # ('normal-3-highrate',['fluorescence_normal-3-highrate.txt.desc.csv-graph2.graphml','network_normal-3-highrate.txt','networkPositions_normal-3-highrate.txt']),
    # ('normal-4',['fluorescence_normal-4.txt.desc.csv-graph2.graphml','network_normal-4.txt','networkPositions_normal-4.txt']),
    # ('normal-4-lownoise',['fluorescence_normal-4-lownoise.txt.desc.csv-graph2.graphml','network_normal-4-lownoise.txt','networkPositions_normal-4-lownoise.txt']),
    # ('highcc',['fluorescence_highcc.txt.desc.csv-graph2.graphml','network_highcc.txt','networkPositions_highcc.txt']),
    # ('highcon',['fluorescence_highcon.txt.desc.csv-graph2.graphml','network_highcon.txt','networkPositions_highcon.txt']),
    # ('lowcc',['fluorescence_lowcc.txt.desc.csv-graph2.graphml','network_lowcc.txt','networkPositions_lowcc.txt']),
    # ('lowcon',['fluorescence_lowcon.txt.desc.csv-graph2.graphml','network_lowcon.txt','networkPositions_lowcon.txt']),
    #     ('valid', ['fluorescence_valid.txt.desc.csv-graph2.graphml','','networkPositions_valid.txt']),
    #     ('test', ['fluorescence_test.txt.desc.csv-graph2.graphml','','networkPositions_test.txt'])
# ]

# prep_nets = [
#     ('valid', ['fluorescence_valid.txt.desc.csv-graph2.graphml','','networkPositions_valid.txt']),
#     ('test', ['fluorescence_test.txt.desc.csv-graph2.graphml','','networkPositions_test.txt'])
# ]

train_nets = [
    ('normal-1',['fluorescence_normal-1.txt.desc.csv-graph2.graphml','network_normal-1.txt','networkPositions_normal-1.txt']),
    ('normal-2',['fluorescence_normal-2.txt.desc.csv-graph2.graphml','network_normal-2.txt','networkPositions_normal-2.txt']),
#     ('normal-3',['fluorescence_normal-3.txt.desc.csv-graph2.graphml','network_normal-3.txt','networkPositions_normal-3.txt']),
#     ('normal-3-highrate',['fluorescence_normal-3-highrate.txt.desc.csv-graph2.graphml','network_normal-3-highrate.txt','networkPositions_normal-3-highrate.txt']),
    ('normal-4',['fluorescence_normal-4.txt.desc.csv-graph2.graphml','network_normal-4.txt','networkPositions_normal-4.txt']),
#     ('normal-4-lownoise',['fluorescence_normal-4-lownoise.txt.desc.csv-graph2.graphml','network_normal-4-lownoise.txt','networkPositions_normal-4-lownoise.txt']),
#     ('highcc',['fluorescence_highcc.txt.desc.csv-graph2.graphml','network_highcc.txt','networkPositions_highcc.txt']),
#     ('highcon',['fluorescence_highcon.txt.desc.csv-graph2.graphml','network_highcon.txt','networkPositions_highcon.txt']),
#     ('lowcc',['fluorescence_lowcc.txt.desc.csv-graph2.graphml','network_lowcc.txt','networkPositions_lowcc.txt']),
#     ('lowcon',['fluorescence_lowcon.txt.desc.csv-graph2.graphml','network_lowcon.txt','networkPositions_lowcon.txt']),
]

test_nets = [
    ('valid',['fluorescence_valid.txt.desc.csv-graph2.graphml','','networkPositions_valid.txt']),
    ('test',['fluorescence_test.txt.desc.csv-graph2.graphml','','networkPositions_test.txt'])
]

_cols = [
    ['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14','e15','d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15'],
    # ['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15'],
    # ['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14','e15'],
    # ['d1','d2','d4','d8']
]

model_file = '/Users/dan/dev/datasci/kaggle/connectomix/out/model2.pkl'

in_dir = '/Users/dan/dev/datasci/kaggle/connectomix/'

def time_remaining(tot_iters, cur_iter, total_dur):
    avg = total_dur/((cur_iter if cur_iter != 0 else 1)*1.0)
    rem = (tot_iters - cur_iter) * avg
    return avg/1000, rem/1000

def prepare(in_dir, nets):
    print('prepare...')

    for t in nets:
        k = t[0]
        v = t[1]
        print(k + ': processing net at ' + str(datetime.now()))
        out_dir = in_dir + k + '/out/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        G = nx.read_graphml(out_dir + v[0])
        n_neurons = len(G.nodes())
        neurons = G.nodes()

        positions = pd.read_table(in_dir + k + '/' + v[2], sep=',', header=None, names=['x','y'])
        positions.index = range(1, len(positions)+1)

        # create dataframe with predictions
        cols = ['i','j','e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11','e12','e13','e14','e15','d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15']
        cols_key = cols[0:2]

        idx_tup = list(itertools.product(neurons, neurons))

        print(k + ': calculating neuron means and deltas...')
        # calculate means and std of each neuron and their connections
        start = time.time() * 1000
        n_ct = 0
        node_means = {}
        for n in neurons:
            n_ct += 1
            if n_ct % 100 == 0:
                cur_time = time.time() * 1000
                avg, rem = time_remaining(n_neurons, n_ct, cur_time - start )
                print(k + ': n iter ' + str(n_ct) + ' | avg=' + str(avg) + ' | rem=' + str(rem))

            nedges = G[n]
            node_means[n] = {}
            col_arrs = []
            for w in range(0, 16): # the number of columns to extract data for
                col_arrs.append([])
            for ekey in nedges.keys():
                edge = nedges[ekey]
                for w in range(0, 16):
                    tstr = 'e' + str(w)
                    col_arrs[w].append(edge[tstr] if tstr in edge else 0)
            for w in range(0, 16):
                tstr = 'e' + str(w)
                node_means[n]['mean' + tstr] = np.mean(col_arrs[w])
                node_means[n]['std' + tstr] = np.std(col_arrs[w])

        # move the edge stats (raw and relative to std) to the DF
        edges = G.edges()
        n_edges = len(edges)
        edge_matrix = np.zeros((n_edges, len(cols)))
        print(k + ': creating edge matrix for # edges = ' + str(n_edges))
        start = time.time() * 1000
        e_ct = 0
        for e in idx_tup:
            e_ct += 1
            if e_ct % 10000 == 0:
                cur_time = time.time() * 1000
                avg, rem = time_remaining(n_edges, e_ct, cur_time - start )
                print(k + ': e iter ' + str(e_ct) + ' | avg=' + str(avg) + ' | rem=' + str(rem))

            nmeans = node_means[e[0]]

            row = np.zeros(32)
            for w in range(0, 16):
                tstr = 'e' + str(w)
                row[w] = G[e[0]][e[1]][tstr] if tstr in G[e[0]][e[1]] else 0
                m = nmeans['mean' + tstr]
                s = nmeans['std' + tstr]
                row[w+16] = (row[w] - m)/s if s > 0 else 0

            edge_matrix[e_ct-1][0] = e[0]
            edge_matrix[e_ct-1][1] = e[1]
            edge_matrix[e_ct-1][2::] = row

        dfm = pd.DataFrame(edge_matrix, columns=cols)
        dfm[['i','j']].astype(np.int)
        dfm.index = pd.MultiIndex.from_tuples(idx_tup)
        dfm.to_csv(out_dir + v[0] + '.test.csv', index=True, header=True)


def evaluate(in_dir, nets):
    print('evaluate...')

    best_auc = 0.0

    X_ref = []
    y_ref = []

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

            if k == 'normal-1':
                X_ref = X
                y_ref = y

            scaler = pre.StandardScaler()
            # scaled = scaler.fit_transform(X)
            # X = pd.DataFrame(scaled, columns=X.columns)

            # xval
            clf = LogisticRegression(C=1,penalty='l1', class_weight={0:1,1:8})
            # clf = RandomForestClassifier(n_estimators=10, max_features=.5)
            scores = sl.cross_validation.cross_val_score(clf, X, y, scoring='roc_auc')
            mscores = np.mean(scores)
            print(k + ': xval scores=' + str(scores) + ' | mean=' + str(mscores))
            if mscores > best_auc:
                best_auc = mscores

            # test on ref
            # clf.fit(X, y)
            # z = clf.predict_proba(X_ref)
            # # this is probably a crappy way to do this
            # probs = []
            # for r in z:
            #     probs.append(r[1])
            # fpr, tpr, t = sl.metrics.roc_curve(y_ref, probs )
            # auc = sl.metrics.auc(fpr, tpr)
            # print (k + ' roc auc ref: ' + str(auc))
            # if auc > best_auc:
            #     best_auc = auc


            # print('learning curve: ' + str(learning_curve(clf, X, y, range(500,30500,10000))))

            # single classification
            # X_train,X_test,y_train,y_test = train_test_split(X, y)
            # print(k + ' y_train ' + str(len(y_train)) + '|' + str(np.sum(y_train)))
            # print(k + ' y_test ' + str(len(y_test)) + '|' + str(np.sum(y_test)))
            # clf.fit(X_train,y_train)
            # z = clf.predict(X_test)
            # print (k + ' conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test,z)))
            # z = clf.predict_proba(X_test)
            # # this is probably a crappy way to do this
            # probs = []
            # for r in z:
            #     probs.append(r[1])
            # fpr, tpr, t = sl.metrics.roc_curve(y_test, probs )
            # auc = sl.metrics.auc(fpr, tpr)
            # print (k + ' roc auc 2: ' + str(auc))
            # if auc > best_auc:
            #     best_auc = auc


            # logistic regression
            # for C in [.01,.1,1,10]:
            #     for p in ['l1','l2']:
            #         for w in [1,2,4,8]:
            #             print (k + ': evaluating C=' + str(C) + ' p=' + p + ' w=' + str(w))
            #             clf = LogisticRegression(C=C,penalty=p, class_weight={0:1,1:w})
            #             scores = sl.cross_validation.cross_val_score(clf, X, y, scoring='roc_auc')
            #             mscores = np.mean(scores)
            #             print(k + ': xval scores=' + str(scores) + ' | mean=' + str(mscores))
            #             if mscores > best_auc:
            #                 best_auc = mscores


            # random forest
            # for e in [5,10,15,25]:
            #     for c in ['gini','entropy']:
            #         for f in [.25,.5,.75,1.0]:
            #             print (k + ': evaluating e=' + str(e) + ' c=' + c + ' f=' + str(f))
            #             clf = RandomForestClassifier(n_estimators=e,criterion=c, max_features=f)
            #             scores = sl.cross_validation.cross_val_score(clf, X, y, scoring='roc_auc')
            #             mscores = np.mean(scores)
            #             print(k + ': xval scores=' + str(scores) + ' | mean=' + str(mscores))
            #             if mscores > best_auc:
            #                 best_auc = mscores

            # gradient boost
            for e in [50,100,200]:
                    for f in [.5,.75,1.0]:
                        print (k + ': evaluating e=' + str(e) + ' f=' + str(f))
                        clf = GradientBoostingClassifier(n_estimators=e, max_features=f, verbose=True)
                        scores = sl.cross_validation.cross_val_score(clf, X, y, scoring='roc_auc')
                        mscores = np.mean(scores)
                        print(k + ': GB xval scores=' + str(scores) + ' | mean=' + str(mscores))
                        if mscores > best_auc:
                            best_auc = mscores


        print ('done; best_auc=' + str(best_auc))

def train(in_dir, nets):
    print('train...')

    dft = []

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

        if len(dft) == 0:
            dft = dfj
        else:
            dft = dft.append(dfj)

    # now train
    scaler = pre.StandardScaler()

    dft = dft[dft['i'] != dft['j']]

    cols = _cols[0]
    X = dft[cols]
    y = dft['act']

    # shouldn't be needed since ranges are modest
    # scaled = scaler.fit_transform(X)
    # X = pd.DataFrame(scaled, columns=X.columns)

    X_train,X_test,y_train,y_test = train_test_split(X, y)
    print(k + ' y_train ' + str(len(y_train)) + '|' + str(np.sum(y_train)))
    print(k + ' y_test ' + str(len(y_test)) + '|' + str(np.sum(y_test)))

    print(k + ': fitting model')
    # per xval on all features: C=1, p=l1, w=8
    # clf = LogisticRegression(C=1,penalty='l1', class_weight={0:1,1:8})
    clf = GradientBoostingClassifier(n_estimators=200, max_features=.75, verbose=True)
    clf.fit(X,y)
    s = pickle.dumps(clf)
    f = open(model_file, 'w')
    f.write(s)
    f.close()

    f = open(model_file, 'r')
    s2 = f.read()
    clf2 = pickle.loads(s2)
    z = clf2.predict(X_test)
    # print (k + ' roc auc: ' + str(roc_auc_score(y_test,z)))
    print (k + ' conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test,z)))
    # print (k + ' class report:\n' + str(sl.metrics.classification_report(y_test,z)))
    z = clf2.predict_proba(X_test)

    # this is probably a crappy way to do this
    probs = []
    for r in z:
        probs.append(r[1])
    fpr, tpr, t = sl.metrics.roc_curve(y_test, probs )
    print (k + ' roc auc 2: ' + str(sl.metrics.auc(fpr, tpr)))
    # score to beat: 0.913357595165

    # scores = sl.cross_validation.cross_val_score(clf2, X, y, scoring='roc_auc')
    # print(k + ': xval scores=' + str(scores) + ' | mean=' + str(np.mean(scores)))
    return scaler

def predict(in_dir, model_file, nets, scaler=None):
    print('predict...')

    res = np.array([])

    for t in nets:
        k = t[0]
        v = t[1]
        print(k + ': predicting at ' + str(datetime.now()))
        out_dir = in_dir + k + '/out/'

        # read tables
        dft = pd.read_table(out_dir + '/' + v[0] + '.test.csv', sep=',', index_col=[0,1])

        cols = _cols[0]
        X = dft[cols]

        # scaled = scaler.transform(X)
        # X = pd.DataFrame(scaled, columns=X.columns)

        f = open(model_file, 'r')
        s2 = f.read()
        clf2 = pickle.loads(s2)
        preds = clf2.predict_proba(X)
        probs = []
        for r in preds:
            probs.append(r[1])

        X['key'] = dft[['i','j']].apply(lambda x: k + '_' + str((int)(x['i'])) + '_' + str((int)(x['j'])), axis=1)
        X['pred'] = probs
        if len(res) == 0:
            res = X[['key', 'pred']].values
        else:
            res = np.concatenate([res, X[['key', 'pred']].values])
        print(k + ': done predicting; k size=' + str(len(X))+ ' | res size=' + str(len(res)))

    print('writing final output')
    df = pd.DataFrame(res,columns=['NET_neuronI_neuronJ','Strength'])
    df.to_csv(in_dir + '/out/predictions2.csv', index=False)
    print('done; num rows=' + str(len(df)))

# prepare(in_dir, prep_nets)

# evaluate(in_dir, train_nets)

scaler = train(in_dir, train_nets)

predict(in_dir, model_file, test_nets, scaler)


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
