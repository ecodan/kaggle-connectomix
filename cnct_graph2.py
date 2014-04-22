__author__ = 'dan'

'''
Step 2 in pipeline

This is the second approach.  Basically it counts complex potential connectivity
between neurons in the current time frame and up to 3 frames back and tracks that in a 2**4 matrix
(flattened to a 4 bit binary number).  For example, in I -> J if I fired current and n-2 frames and J fired current,
this would increment I(0101) -> J or I(5) -> J.

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

# nets = [
#     ('normal-1','fluorescence_normal-1.txt.desc.csv'),
#     ]

# nets = [
#     ('normal-2','fluorescence_normal-2.txt.desc.csv'),
#     ('normal-3-highrate','fluorescence_normal-3-highrate.txt.desc.csv'),
#     ('normal-3','fluorescence_normal-3.txt.desc.csv'),
#     ('normal-4','fluorescence_normal-4.txt.desc.csv'),
#     ('normal-4-lownoise','fluorescence_normal-4-lownoise.txt.desc.csv'),
#     ('highcc','fluorescence_highcc.txt.desc.csv'),
#     ('lowcc','fluorescence_lowcc.txt.desc.csv'),
#     ('highcon','fluorescence_highcon.txt.desc.csv'),
#     ('lowcon','fluorescence_lowcon.txt.desc.csv')
# ]

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

            # look back holistically over last N frames and if there is a possible impact track as a unidirectional effect
            if t >= 3:
                # get a slice of 4 time frames starting at the current
                window = n_act.iloc[t-3:t+1].values
                # get a list of columns with values other than zero
                wsum = np.sum(window, axis=0)
                act_window = np.array(range(0,neuron_ct))
                act_window = act_window[wsum > 0]
                if len(act_window) == 0:
                    # nothing here
                    continue
                if len(act_window) > (neuron_ct*.2):
                    # ignore burst period windows
                    continue
                activated_current = np.array(range(0,neuron_ct))
                activated_current = activated_current[window[3] > 0]
                for i in activated_current:
                    for j in act_window:
                        if i == j: continue
                        # print('process edge ' + str(i+1) + ',' + str(j+1))
                        edge = G[j+1][i+1] # process entropy from J to I
                        if 'e1' not in edge:
                            for e in range(0,16):
                                edge['e'+str(e)] = 0
                        _ent = array_to_int(window[::,j])
                        edge['e' + str(_ent)] += 1



        print(k + ': DIAG nodes=' + str(len(G.nodes())))
        print(k + ': DIAG edges=' + str(len(G.edges())))
        print(k + ': DIAG thr cts=' + str(totals))

        nx.write_graphml(G, out_dir + v + '-graph2.graphml')

    return



nw_graph(in_dir, nets)