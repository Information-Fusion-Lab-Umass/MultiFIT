import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pickle

class RNN_osaka(nn.Module):
    def __init__(self, params):
        super(RNN_osaka, self).__init__()
        
        self.bilstm_flag = params['bilstm_flag']
        self.hidden_dim_fast = params['hidden_dim_fast']
        self.hidden_dim_slow = params['hidden_dim_slow']
        self.dropout = params['dropout']
        self.layers = params['layers']
        self.tagset_size = params['tagset_size']
        self.attn_category = params['attn_category']
        self.num_features_slow = params['num_features_slow']
        self.num_features_fast = params['num_features_fast']
        self.imputation_layer_dim_op = params['imputation_layer_dim_op']
        self.selected_feats = params['selected_feats']
        self.slow_features_indexes = params['slow_features_indexes']
        self.fast_features_indexes = params['fast_features_indexes']
        
        self.imputation_layer_dim_in_fast = (self.selected_feats+1)*4
        self.imputation_layer_dim_in_slow = (self.selected_feats+1)*4
        
        self.input_dim_fast = self.num_features_fast * self.imputation_layer_dim_op
        self.input_dim_slow = self.num_features_slow * self.imputation_layer_dim_op
#         self.hidden_dim = 2*self.input_dim
        
        
        self.dict_selected_feats_fast = {}
        self.dict_selected_feats_slow  = {}
        if(params['feats_provided_flag']==True):
            print "Oooh! the support features are provided!"
            self.dict_selected_feats_fast = pickle.load(open(params['path_selected_feats_dict_fast'],'rb'))
            self.dict_selected_feats_slow = pickle.load(open(params['path_selected_feats_dict_slow'],'rb'))
            for each_ind in range(self.num_features_fast):
                self.dict_selected_feats_fast[each_ind] = [each_ind]+self.dict_selected_feats_fast[each_ind]
            for each_ind in range(self.num_features_slow):
                self.dict_selected_feats_slow[each_ind] = [each_ind]+self.dict_selected_feats_slow[each_ind]
        else:
            for each_ind in range(self.num_features):
                all_feats = range(self.num_features)
                all_feats.remove(each_ind)
                random.shuffle(all_feats)
                self.dict_selected_feats[each_ind] = [each_ind] + all_feats[:self.selected_feats]
        
#         self.LL = nn.Linear(self.len_features, self.input_dim)
        self.imputation_layer_in_fast = [nn.Linear(self.imputation_layer_dim_in_fast,self.imputation_layer_dim_op).cuda() for x in range(self.num_features_fast)]

        self.imputation_layer_in_slow = [nn.Linear(self.imputation_layer_dim_in_slow,self.imputation_layer_dim_op).cuda() for x in range(self.num_features_slow)]
        self.imputation_layer_in_fast = nn.ModuleList(self.imputation_layer_in_fast)
        self.imputation_layer_in_slow = nn.ModuleList(self.imputation_layer_in_slow)
#         self.imputation_layer_op = nn.Linear(self.imputation_layer_dim, 1)
        
        if(self.bilstm_flag):
            self.lstm_fast = nn.LSTM(self.input_dim_fast, self.hidden_dim_fast/2, num_layers = self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
            self.lstm_slow = nn.LSTM(self.input_dim_slow, self.hidden_dim_slow/2, num_layers = self.layers,
                                bidirectional=True, batch_first=True, dropout=self.dropout)
        else:
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers = self.layers, 
                                bidirectional=False, batch_first=True, dropout=self.dropout)
        
        self.hidden2tag = nn.Linear(self.hidden_dim_fast+self.hidden_dim_slow, self.tagset_size)
        
        if(self.attn_category == 'dot'):
            print "Dot Attention is being used!"
            self.attn_fast = DotAttentionLayer(self.hidden_dim_fast).cuda()
            self.attn_slow = DotAttentionLayer(self.hidden_dim_slow).cuda()

    
    def init_hidden(self, batch_size, hidden_dim):
    # num_layes, minibatch size, hidden_dim
        if(self.bilstm_flag):
            return (autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                             batch_size,
                                                             hidden_dim/2).fill_(0)),
                   autograd.Variable(torch.cuda.FloatTensor(self.layers*2,
                                                            batch_size,
                                                            hidden_dim/2).fill_(0)))
        else:
            return (autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                             batch_size,
                                                             hidden_dim).fill_(0)),
                   autograd.Variable(torch.cuda.FloatTensor(self.layers,
                                                            batch_size,
                                                            hidden_dim).fill_(0)))
    
    def forward(self, data, id_):
#         features = self.LL(features)
        batch_size = 1
    
        if(data[id_][3]==0):
#             print "data"
            features_fast = self.get_imputed_feats(data[id_][0], data[id_][1], data[id_][2],self.dict_selected_feats_fast, self.imputation_layer_in_fast, len(self.fast_features_indexes))
            lenghts_fast = [features_fast.shape[1]]
            lengths_fast = torch.cuda.LongTensor(lenghts_fast)
            lengths_fast = autograd.Variable(lengths_fast)
            packed_fast = pack_padded_sequence(features_fast, lengths_fast, batch_first = True)
            self.hidden_fast = self.init_hidden(batch_size, self.hidden_dim_fast)
            packed_output_fast, self.hidden_fast = self.lstm_fast(packed_fast, self.hidden_fast)
            lstm_out_fast = pad_packed_sequence(packed_output_fast, batch_first=True)[0]
            if(self.attn_category=='dot'):
                pad_attn_fast = self.attn_fast((lstm_out_fast, torch.cuda.LongTensor(lengths_fast)))        
        else:
            pad_attn_fast = torch.cuda.FloatTensor(np.zeros([1,self.hidden_dim_fast]))
        
        if(data[id_][7]==0):
#             print "data"
            features_slow = self.get_imputed_feats(data[id_][4], data[id_][5], data[id_][6],self.dict_selected_feats_slow, self.imputation_layer_in_slow, len(self.slow_features_indexes))
            lenghts_slow = [features_slow.shape[1]]
            lengths_slow = torch.cuda.LongTensor(lenghts_slow)
            lengths_slow = autograd.Variable(lengths_slow)
            packed_slow = pack_padded_sequence(features_slow, lengths_slow, batch_first = True)
            self.hidden_slow = self.init_hidden(batch_size, self.hidden_dim_slow)
            packed_output_slow, self.hidden_slow = self.lstm_slow(packed_slow, self.hidden_slow)
            lstm_out_slow = pad_packed_sequence(packed_output_slow, batch_first=True)[0]
            if(self.attn_category=='dot'):
                pad_attn_slow = self.attn_slow((lstm_out_slow, torch.cuda.LongTensor(lengths_slow)))
        else:
            pad_attn_slow = torch.cuda.FloatTensor(np.zeros([1,self.hidden_dim_slow]))
#             print pad_attn_final.shape
#         else:
            # For now this won't work
#             tag_space = self.hidden2tag(lstm_out[:,-1,:])
#         print pad_attn_fast.shape
#         print pad_attn_slow.shape
        pad_attn_final = torch.cat([pad_attn_fast, pad_attn_slow],1)
        tag_space = self.hidden2tag(pad_attn_final)
#         print tag_space
        tag_score = F.log_softmax(tag_space, dim=1)
        return tag_score
    
    def get_imputed_feats(self, feats, flags, time_steps, dict_selected_feats, imputation_layer_in, num_features):
        feats = np.asarray(feats)
        flags = np.asarray(flags)
        # we have time steps as well in this script.
        all_features = []
        input_ = {}
        for feat_ind in range(num_features):
            input_[feat_ind] = []
            feat = feats[:,feat_ind]
            feat_flag = flags[:,feat_ind]
            ind_keep = feat_flag==0
            ind_missing = feat_flag==1
            if(sum(ind_keep)>0):
                avg_val = np.mean(feat[ind_keep])
            else:
                avg_val = 0.0
            last_val_observed = avg_val
            last_t_observed = 0
            # actual time step is being used here.
#             delta_t = -1
            for ind, each_flag in enumerate(feat_flag):
                if(each_flag==1):
                    delta_t = time_steps[ind] - last_t_observed
                    imputation_feat = [last_val_observed, avg_val, 1, delta_t]
                    input_[feat_ind].append(imputation_feat)
                elif(each_flag==0):
                    last_t_observed = time_steps[ind]
                    delta_t = time_steps[ind] - last_t_observed # which would obviously be zero, just to keep things consistent
                    last_val_observed = feat[ind]
                    imputation_feat = [last_val_observed, avg_val, 0, delta_t]
                    input_[feat_ind].append(imputation_feat)
#                 delta_t+=1
        for feat_ind in range(num_features):
            final_feat_list = []
            for ind, each_flag in enumerate(feat_flag):
                imputation_feat = []
                for each_selected_feat in dict_selected_feats[feat_ind]:
                    imputation_feat+=input_[each_selected_feat][ind]
                imputation_feat = autograd.Variable(torch.cuda.FloatTensor(imputation_feat))
                f_= imputation_layer_in[feat_ind](imputation_feat)
                final_feat_list.append(f_)
            final_feat_list = torch.stack(final_feat_list)
            all_features.append(final_feat_list)
        all_features = torch.cat(all_features,1)
        all_features = all_features.unsqueeze(0)
        all_features = autograd.Variable(all_features)
        return all_features
    
    def prepare_batch(self, dict_data, ids):
        labels = []
        for each_id in ids:
            t_label = label_mapping[dict_data[each_id][1]]
            labels.append(t_label)
        features = []
        max_len = 0
        actual_lens = []
        
        for each_id in ids:
            t_features = dict_data[each_id][0]
            features.append(t_features)
            if(len(t_features)>max_len):
                max_len = len(t_features)
            actual_lens.append(len(t_features))

        for ind in range(len(features)):
            features[ind] = features[ind]+[[0 for x in range(21)] for y in range(max_len-len(features[ind]))]

        sorted_inds = np.argsort(actual_lens)
        sorted_inds = sorted_inds[::-1]

        sorted_lens = []
        sorted_features = []
        sorted_labels = []
        ind_cnt = 0
        for ind in sorted_inds:
            sorted_lens.append(actual_lens[ind])
            sorted_features.append(features[ind])
            sorted_labels.append(labels[ind])

        sorted_features = torch.cuda.FloatTensor(sorted_features)
        sorted_features = autograd.Variable(sorted_features)

        sorted_labels = torch.cuda.LongTensor(sorted_labels)
        sorted_labels = autograd.Variable(sorted_labels)

        return sorted_features, sorted_labels, sorted_lens


class DotAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input):
        """
        input: (unpacked_padded_output: batch_size x seq_len x hidden_size, lengths: batch_size)
        """
        inputs, lengths = input
        batch_size, max_len, _ = inputs.size()
        flat_input = inputs.contiguous().view(-1, self.hidden_size)
        logits = self.W(flat_input).view(batch_size, max_len)
        alphas = F.softmax(logits, dim=1)

        # computing mask
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0).cuda()
        mask = autograd.Variable((idxes<lengths.unsqueeze(1)).float())

        alphas = alphas * mask
        # renormalize
        alphas = alphas / torch.sum(alphas, 1).view(-1, 1)
        output = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)
        return output