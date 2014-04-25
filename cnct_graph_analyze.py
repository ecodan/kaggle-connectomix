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
from datetime import datetime

import pybrain as pb
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer


# prep_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt'])]
# prep_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC02inh.txt.desc.csv-graph.graphml','network_iNet1_Size100_CC02inh.txt','networkPositions_iNet1_Size100_CC02inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC03inh.txt.desc.csv-graph.graphml','network_iNet1_Size100_CC03inh.txt','networkPositions_iNet1_Size100_CC03inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC04inh.txt.desc.csv-graph.graphml','network_iNet1_Size100_CC04inh.txt','networkPositions_iNet1_Size100_CC04inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC05inh.txt.desc.csv-graph.graphml','network_iNet1_Size100_CC05inh.txt','networkPositions_iNet1_Size100_CC05inh.txt']),
#              ('small',['fluorescence_iNet1_Size100_CC06inh.txt.desc.csv-graph.graphml','network_iNet1_Size100_CC06inh.txt','networkPositions_iNet1_Size100_CC06inh.txt'])]
# test_nets = [('small',['fluorescence_iNet1_Size100_CC01inh.txt.desc.csv-graph.graphml','network_iNet1_Size100_CC01inh.txt','networkPositions_iNet1_Size100_CC01inh.txt'])]
# train_nets = prep_nets

# prep_nets = [
#     ('normal-1',['fluorescence_normal-1.txt.desc.csv-graph.graphml','network_normal-1.txt','networkPositions_normal-1.txt'])
# ]
# prep_nets = [
#     ('valid', ['fluorescence_valid.txt.desc.csv-graph.graphml','','networkPositions_valid.txt']),
#     ('test', ['fluorescence_test.txt.desc.csv-graph.graphml','','networkPositions_test.txt'])
# ]
train_nets = [
    ('normal-1',['fluorescence_normal-1.txt.desc.csv-graph.graphml','network_normal-1.txt','networkPositions_normal-1.txt'])
]
test_nets = [
    ('valid',['fluorescence_valid.txt.desc.csv-graph.graphml','','networkPositions_valid.txt']),
    ('test',['fluorescence_test.txt.desc.csv-graph.graphml','','networkPositions_test.txt'])
]

model_file = '/Users/dan/dev/datasci/kaggle/connectomix/out/model.pkl'

in_dir = '/Users/dan/dev/datasci/kaggle/connectomix/'

def calc_dist(p1,p2):
    return math.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 )

def learning_curve(clf, X, y, sizes):
    print('starting learning curve analysis...')
    res = []
    for i in sizes:
        print('starting evaluation of data size ' + str(i))
        X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=i, test_size=i)
        clf.fit(X_train, y_train)
        z = clf.predict_proba(X_train)
        probs = []
        for r in z:
            probs.append(r[1])
        fpr, tpr, t = sl.metrics.roc_curve(y_train, probs )
        train_score = sl.metrics.auc(fpr, tpr)

        z = clf.predict_proba(X_test)
        probs = []
        for r in z:
            probs.append(r[1])
        fpr, tpr, t = sl.metrics.roc_curve(y_test, probs )
        test_score = sl.metrics.auc(fpr, tpr)
        res.append([train_score, test_score])

    return res

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
        positions = pd.read_table(in_dir + k + '/' + v[2], sep=',', header=None, names=['x','y'])
        positions.index = range(1, len(positions)+1)

        # create dataframe with predictions
        cols = ['i','j','1-0','1-1','1-2','1-3','d1-0','d1-1','d1-2','d1-3','1-B','2-0','2-1','2-2','2-3','d2-0','d2-1','d2-2','d2-3','2-B','3-0','3-1','3-2','3-3','d3-0','d3-1','d3-2','d3-3','3-B','pred']
        dfc = pd.DataFrame(np.zeros([len(G.nodes())**2, len(cols)]), columns=cols)
        idx_tup = list(itertools.product(G.nodes(), G.nodes()))
        dfc.index = pd.MultiIndex.from_tuples(idx_tup)
        dfc[['d1-0','d1-1','d1-2','d1-3','d2-0','d2-1','d2-2','d2-3','d3-0','d3-1','d3-2','d3-3']].astype(np.float)
        dfc[['i','j']].astype(np.int)

        # populate the is and js
        for i in G.nodes():
            for j in G.nodes():
                dfc['i'].ix[(i,j)] = i
                dfc['j'].ix[(i,j)] = j

        # calculate euclidean distance
        print(k + ': calculating distances...')
        dfc['dist'] = 0.0
        for i in range(1, len(positions)+1):
            for j in range(1, len(positions)+1):
                if i == j: continue
                dfc['dist'].ix[(str(i),str(j))] = calc_dist(positions.ix[i], positions.ix[j])

        degrees = 3

        print(k + ': calculating neuron means and deltas...')
        # calculate means and std of each neuron and their connections
        node_means = {}
        for n in G.nodes():
            nedges = G[n]
            node_means[n] = {}
            for thr in [1,2,3]:
                deg = []
                for w in range(0, degrees+1):
                    deg.append([])
                for k in nedges.keys():
                    edge = nedges[k]
                    for w in range(0, degrees+1):
                        tstr = str(thr) + '-' + str(w)
                        deg[w].append(edge[tstr] if tstr in edge else 0)
                for w in range(0, degrees+1):
                    tstr = str(thr) + '-' + str(w)
                    node_means[n]['mean' + tstr] = np.mean(deg[w])
                    node_means[n]['std' + tstr] = np.std(deg[w])

        # move the edge stats (raw and relative to std) to the DF
        for e in G.edges():
            nmeans = node_means[e[0]]
            for thr in [1,2,3]:
                # first do blasts
                tstr = str(thr) + '-B'
                dfc[tstr].ix[e] = G[e[0]][e[1]][tstr] if tstr in G[e[0]][e[1]] else 0
                # now calc all degrees
                for w in range(0, degrees+1):
                    tstr = str(thr) + '-' + str(w)
                    dfc[tstr].ix[e] = G[e[0]][e[1]][tstr] if tstr in G[e[0]][e[1]] else 0
                    m = nmeans['mean' + tstr]
                    s = nmeans['std' + tstr]
                    dfc['d'+tstr].ix[e] = (dfc[tstr].ix[e] - m)/s if s > 0 else 0

        dfc.to_csv(out_dir + v[0] + '.test.csv', index=True, header=True)


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

        # quick classifier
        # X = dft[['d1-0']]
        # X = dft[['d1-0','d1-1','d1-2','d1-3','dist']]
        # X = dft[['1-0','1-1','1-2','1-3','d1-0','d1-1','d1-2','d1-3','1-B','2-0','2-1','2-2','2-3','d2-0','d2-1','d2-2','d2-3','2-B','3-0','3-1','3-2','3-3','d3-0','d3-1','d3-2','d3-3','3-B','dist']]
        X = dft[['d1-0','d1-1','d1-2','d1-3','d2-0','d2-1','d2-2','d2-3','d3-0','d3-1','d3-2','d3-3','dist']]
        # X = dft[['d1-0','d1-1','d1-2','d1-3']]
        y = dft['act']

        # xval
        clf = LogisticRegression(C=1,penalty='l1', class_weight={0:1,1:2})
        # clf = RandomForestClassifier(n_estimators=10, max_features=.5)
        scores = sl.cross_validation.cross_val_score(clf, X, y, scoring='roc_auc')
        mscores = np.mean(scores)
        print(k + ': xval scores=' + str(scores) + ' | mean=' + str(mscores))
        if mscores > best_auc:
            best_auc = mscores

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

        # neural net
        X_train,X_test,y_train,y_test = train_test_split(X, y)
        net = buildNetwork(X_train.shape[1], X_train.shape[1] * 2, 1, bias=True, hiddenclass=TanhLayer)
        ds = SupervisedDataSet(X_train.shape[1], 1)
        ds.setField('input', X_train)
        ds.setField('target', y_train.reshape(-1, 1))
        trainer = BackpropTrainer(net, ds)
        trainer.trainUntilConvergence( verbose = True, validationProportion = 0.25, maxEpochs = 50, continueEpochs = 10 )

        ds = SupervisedDataSet(X_test.shape[1], 1)
        ds.setField('input', X_test)
        ds.setField('target', y_test.reshape(-1,1))
        z = net.activateOnDataset(ds)
        print (k + ' NN results:\n' + str(list(z)))
        fpr, tpr, t = sl.metrics.roc_curve(y_test, z )
        print (k + ': roc auc: ' + str(sl.metrics.auc(fpr, tpr)))


        print ('done; best_auc=' + str(best_auc))

def train(in_dir, model_file, nets):
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
        # dft = dfj[dfj['0-1'] != 1]
        dft = dfj[dfj['i'] != dfj['j']]
        # X = dft[['d1-0','d1-1','d2-0','d3-0']]
        # X = dft[['d1-0','d1-1','d1-2','d1-3']]
        # X = dft[['d1-0','d1-1','d1-2','d1-3','d2-0','d2-1','d2-2','d2-3','d3-0','d3-1','d3-2','d3-3','dist']]
        X = dft[['1-0','1-1','1-2','1-3','d1-0','d1-1','d1-2','d1-3','1-B','2-0','2-1','2-2','2-3','d2-0','d2-1','d2-2','d2-3','2-B','3-0','3-1','3-2','3-3','d3-0','d3-1','d3-2','d3-3','3-B','dist']]
        # X = dft[['1-0','1-1','d1-0','d1-1','1-B','2-0','d2-0','2-B','3-0','d3-0','3-B','dist']]

        # scaled = scaler.fit_transform(X)
        # X = pd.DataFrame(scaled, columns=X.columns)

        y = dft['act']
        X_train,X_test,y_train,y_test = train_test_split(X, y)
        print(k + ' y_train ' + str(len(y_train)) + '|' + str(np.sum(y_train)))
        print(k + ' y_test ' + str(len(y_test)) + '|' + str(np.sum(y_test)))

        print(k + ': fitting model')
        # per xval on all features: C=1, p=l1, w=8
        clf = LogisticRegression(C=1,penalty='l1', class_weight={0:1,1:8})
        # clf = SVC(C=1, kernel='poly')
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

        scores = sl.cross_validation.cross_val_score(clf2, X, y, scoring='roc_auc')
        print(k + ': xval scores=' + str(scores) + ' | mean=' + str(np.mean(scores)))
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

        # X = dft[['d1-0','d1-1','d1-2','d1-3','dist']]
        # X = dft[['d1-0','d1-1','d1-2','d1-3','d2-0','d2-1','d2-2','d2-3','d3-0','d3-1','d3-2','d3-3','dist']]
        X = dft[['1-0','1-1','1-2','1-3','d1-0','d1-1','d1-2','d1-3','1-B','2-0','2-1','2-2','2-3','d2-0','d2-1','d2-2','d2-3','2-B','3-0','3-1','3-2','3-3','d3-0','d3-1','d3-2','d3-3','3-B','dist']]

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
    df.to_csv(in_dir + '/out/predictions.csv', index=False)
    print('done; num rows=' + str(len(df)))

# prepare(in_dir, prep_nets)

evaluate(in_dir, train_nets)

# scaler = train(in_dir, model_file, train_nets)
#
# predict(in_dir, model_file, test_nets, scaler)


# still to try:
# - neural net
# - KNN - default config got .732
# - adjusting features
# - different graph model treatment of burst periods
# - ensemble between correlation and graph approach
# - create 8x8 pre-active probability matrix on each connection
