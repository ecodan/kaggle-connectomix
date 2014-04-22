__author__ = 'dan'

'''
Step 1 in pipeline

Input: flourescence files

Output:
1) diff'd time series (delta between each two time frames)
2) descretized time series (deltas converted to binary with threshold N)

'''

import pandas as pd
import numpy as np
import os
from datetime import datetime

# nets = [('small','fluorescence_iNet1_Size100_CC01inh.txt'),
#         ('small','fluorescence_iNet1_Size100_CC02inh.txt'),
#         ('small','fluorescence_iNet1_Size100_CC03inh.txt'),
#         ('small','fluorescence_iNet1_Size100_CC04inh.txt'),
#         ('small','fluorescence_iNet1_Size100_CC05inh.txt'),
#         ('small','fluorescence_iNet1_Size100_CC06inh.txt')]
# nets = [('valid','fluorescence_valid.txt'),
#           ('test','fluorescence_test.txt')]
nets = [('normal-1','fluorescence_normal-1.txt'),
        ('normal-2','fluorescence_normal-2.txt'),
        ('normal-3-highrate','fluorescence_normal-3-highrate.txt'),
        ('normal-3','fluorescence_normal-3.txt'),
        ('normal-4','fluorescence_normal-4.txt'),
        ('normal-4-lownoise','fluorescence_normal-4-lownoise.txt'),
        ('highcc','fluorescence_highcc.txt'),
        ('lowcc','fluorescence_lowcc.txt'),
        ('highcon','fluorescence_highcon.txt'),
        ('lowcon','fluorescence_lowcon.txt')
        ]
in_dir = '/Users/dan/dev/datasci/kaggle/connectomix/'
threshold = 0.12

def munge(in_dir, nets, threshold=0.12):

    for t in nets:
        k = t[0]
        v = t[1]
        out_dir = in_dir + k + '/out/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # read flourescence file
        in_file = in_dir + k + '/' + v
        print(k + ': reading file ' + in_file + ' at ' + str(datetime.now()))
        df = pd.read_table(in_file, sep=',', header=None)
        df.columns = range(1, len(df.columns) + 1)

        # diff and output
        print(k + ': calclating diff...')
        dfd = df.diff()
        print(k + ': writing diff file ' + out_dir + v + '.diff.csv')
        dfd.to_csv(out_dir + v + '.diff.csv', index=False, header=False)

        # discretize and output
        print (k + ': discretizing...')
        df_desc = dfd.applymap(lambda x: 3 if x > 3 * threshold else 2 if x > 2 * threshold else 1 if x > threshold else 0)
        print(k + ': writing desc file ' + out_dir + v + '.desc.csv')
        df_desc.to_csv(out_dir + v + '.desc.csv', index=False, header=False)


munge(in_dir, nets, threshold)