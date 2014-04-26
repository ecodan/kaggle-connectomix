__author__ = 'dan'

'''
Step 2 in pipeline

This is the first approach I took.  Basically it counts single occurances of potential connectivity
between neurons in the current time frame and up to N frames back.

Input: the descretized file
Output: a graphml file with info about each directed edge
'''

import networkx as nx
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
import time


# nets = [('small','fluorescence_iNet1_Size100_CC01inh.txt.desc.csv')]
# nets = [('small','fluorescence_iNet1_Size100_CC01inh.txt.desc.csv'),
#         ('small','fluorescence_iNet1_Size100_CC02inh.txt.desc.csv'),
#         ('small','fluorescence_iNet1_Size100_CC03inh.txt.desc.csv'),
#         ('small','fluorescence_iNet1_Size100_CC04inh.txt.desc.csv'),
#         ('small','fluorescence_iNet1_Size100_CC05inh.txt.desc.csv'),
#         ('small','fluorescence_iNet1_Size100_CC06inh.txt.desc.csv')]

nets = [('valid','fluorescence_valid.txt.desc.csv'),
        ('test','fluorescence_test.txt.desc.csv')]

nets = [
#     ('normal-1','fluorescence_normal-1.txt.desc.csv'),
    ('normal-2','fluorescence_normal-2.txt.desc.csv'),
    ('normal-3-highrate','fluorescence_normal-3-highrate.txt.desc.csv'),
    ('normal-3','fluorescence_normal-3.txt.desc.csv'),
    ('normal-4','fluorescence_normal-4.txt.desc.csv'),
    ('normal-4-lownoise','fluorescence_normal-4-lownoise.txt.desc.csv'),
    ('highcc','fluorescence_highcc.txt.desc.csv'),
    ('lowcc','fluorescence_lowcc.txt.desc.csv'),
    ('highcon','fluorescence_highcon.txt.desc.csv'),
    ('lowcon','fluorescence_lowcon.txt.desc.csv')
    ]

in_dir = '/Users/dan/dev/datasci/kaggle/connectomix/'


def array_to_int(a):
    ret = 0
    for i in range(0, len(a)):
        exp = len(a)-1-i
        ret += 2**exp if a[i] != 0 else 0
    return ret

def time_remaining(tot_iters, cur_iter, total_dur):
    avg = total_dur/((cur_iter if cur_iter != 0 else 1)*1.0)
    rem = (tot_iters - cur_iter) * avg
    return avg/1000, rem/1000


def nw_graph(in_dir, nets, lookforward_pers=3):

    for t in nets:
        k = t[0]
        v = t[1]
        print(k + ': starting network analyze at ' + str(datetime.now()))
        G = nx.DiGraph()

        out_dir = in_dir + k + '/out/'

        print('reading discretized file')
        n_act = pd.read_table(in_dir + k + '/out/' + v, sep=',', header=None)

        # set the columns headers to 1-based series
        neuron_ct = len(n_act.columns)
        n_act.columns = range(1, neuron_ct+1)

        # add nodes and all possible edges
        G.add_nodes_from(range(1,neuron_ct+1))
        p = itertools.product(G.nodes(), G.nodes())
        G.add_edges_from(p)

        activated_lasts = [[],[],[]]

        totals = [0l,0l,0l]

        print(k + ': starting at ' + str(time.time()))
        start = time.time() * 1000

        # loop through all time periods
        for t in range(0,len(n_act)):
            if t % 1000 == 0:
                cur_time = time.time() * 1000
                avg, rem = time_remaining(len(n_act), t, cur_time - start )
                print(k + ': on time period ' + str(t) + ' | avg=' + str(avg) + ' | rem=' + str(rem))

            # this section is for discreet historical connectivity tracking
            for thr in [1,2,3]:
                # get all of the values above threshold
                activated_current = n_act.columns.values[n_act.iloc[t].values >= thr]

                totals[thr-1] += len(activated_current)

                # get the historical list for this threshold
                activated_last = activated_lasts[thr-1]
                tstr = str(thr)

                # connect all activated (bi-directional)
                p = itertools.product(activated_current, activated_current)
                for tup in p:
                    i = tup[0]
                    j = tup[1]
                    if i == j: continue
                    edge = G[i][j]
                    redge = G[j][i]

                    # if > 20% of neurons fired assume a blast event and clear the history
                    if len(activated_current) > len(G.nodes())*.2:
                        f = tstr + '-B'
                        edge[f] = edge[f] + 1 if f in edge else 1
                        redge[f] = redge[f] + 1 if f in redge else 1
                        activated_last = []
                    else:
                        f = tstr + '-0'
                        edge[f] = edge[f] + 1 if f in edge else 1
                        redge[f] = redge[f] + 1 if f in redge else 1

                # connect with previous periods (uni-directional)
                if (len(activated_last) > 0):
                    degrees = 0
                    for i,e in reversed(list(enumerate(activated_last))):
                        degrees += 1
                        p = itertools.product(e, activated_current)
                        for tup in p:
                            i = tup[0]
                            j = tup[1]
                            if i == j: continue
                            edge = G[i][j]
                            # G.add_edge(i,j)
                            f = tstr + '-' + str(degrees)
                            edge[f] = edge[f] + 1 if f in edge else 1

                # drop the current activated list back one cycle
                activated_last.append(activated_current)
                if (len(activated_last) > lookforward_pers):
                    activated_last.pop(0)
                activated_current = []


        print(k + ': DIAG nodes=' + str(len(G.nodes())))
        print(k + ': DIAG edges=' + str(len(G.edges())))
        print(k + ': DIAG thr cts=' + str(totals))

        nx.write_graphml(G, out_dir + v + '-graph.graphml')

    return



nw_graph(in_dir, nets)