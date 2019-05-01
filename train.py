#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import cPickle as pickle
from random import shuffle
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import os

import sys
sys.path.append('src/')
import Multi_FIT_V as Multi_FIT_V

data = pickle.load(open('data/final_physio_avg_new_split.pkl','rb'))


params = {'bilstm_flag':True,
        'hidden_dim_fast' : 300,
        'hidden_dim_slow' : 200,
        'dropout' : 0.9,
        'layers' : 1,
        'tagset_size' : 2,
        'bilstm_flag' : True,
        'attn_category' : 'dot',
        'num_features_fast' : 26,
        'num_features_slow' : 11,
        'imputation_layer_dim_op':16,
        'selected_feats' : 3,
        'batch_size':1,
        'same_device':False,
        'same_feat_other_device':False,
        'model_name':'MultiRes-',
        'feats_provided_flag':True,
        'path_selected_feats_dict_slow':'../data/dict_selected_feats_physionet_slow',
        'path_selected_feats_dict_fast':'../data/dict_selected_feats_physionet_fast',
        'slow_features_indexes': [0,1,2,3,5,6,16,27,28,31,32],
        'fast_features_indexes': [4,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,29,30,33,34,35,36]}
pickle.dump(params, open('../../Models/config_'+params['model_name']+'.pt','wb'))


model_RNN = Multi_FIT_V.RNN_osaka(params).cuda()
loss_function = nn.NLLLoss()
# optimizer = optim.Adam(model_RNN.parameters(), lr=0.01, weight_decay=0.00005)
optimizer = optim.SGD(model_RNN.parameters(), lr=0.00008, weight_decay=0.00000000002)


mode = 'normal'
if(mode=='normal'):
    feature_ind = 0
    label_ind = -1
    print "NORMAL mode with Flags"

batch_size = 1
save_flag = True
dict_df_prf_mod = {}
print "==x=="*20
print "Data Statistics"
print "Train Data: "+str(len(data['train_ids']))
print "Val Data: "+str(len(data['val_ids']))
print "Test Data: "+str(len(data['test_ids']))
print "Counter of Labels"
print Counter([data['data'][x][-1] for x in data['data'].keys()])
print "==x=="*20


start_epoch = 0
end_epoch = 60
model_name = params['model_name']
for iter_ in range(start_epoch, end_epoch):
    print "=#="*5+str(iter_)+"=#="*5
    total_loss = 0
    preds_train = []
    actual_train = []
    for each_ID in tqdm(data['train_ids']):
        model_RNN.zero_grad()
        tag_scores = model_RNN(data['data'], each_ID)
        
        _, ind_ = torch.max(tag_scores, dim=1)
        preds_train+=ind_.tolist()
        # For this dataset the label is in -2
        curr_labels = [data['data'][each_ID][label_ind]]
        curr_labels = [batchify.label_mapping[x] for x in curr_labels]
        actual_train+=curr_labels
        curr_labels = torch.cuda.LongTensor(curr_labels)
        curr_labels = autograd.Variable(curr_labels)
        
        loss = loss_function(tag_scores, curr_labels.reshape(tag_scores.shape[0]))
        total_loss+=loss.item()

        loss.backward()
        optimizer.step()