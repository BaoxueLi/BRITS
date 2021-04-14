import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import numpy as np
import utils
import argparse
import data_loader

from ipdb import set_trace
from sklearn import metrics

SEQ_LEN = 100
FLAG_IMPUTATION = True

def generate_inf_mask(atten_dim):
    unit_mat = 1 - np.identity(atten_dim)
    mask = torch.from_numpy(unit_mat).cuda()
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



class Model(nn.Module):

    def __init__(self,
        rnn_hid_size, 
        impute_weight, 
        label_weight,
        input_size,
        embed_size=32,
        m_dim =36,
        t_dim = SEQ_LEN,
        T_unit = 5, #measure->heads * T_unit
        M_unit = 10, #time->heads * M_unit
        v_unit = 1, #1->heads * v_unit
        n_heads=8, 
        n_layers=4,
        dropout=0.1,
        mode='time'
        ):
        super(Model, self).__init__()

        # # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(1, n_heads * v_unit, 1)
        self.conv2 = nn.Conv2d(1, n_heads * v_unit, 1)

        # ? [B, T, M, d], d_model = M * d
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=m_dim * n_heads * v_unit, 
                                    nhead=n_heads, dropout=dropout, dim_feedforward=1000)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=n_layers)
        # ? [B, M, T, d], d_model = T * d
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=t_dim * n_heads * v_unit,
                                    nhead=n_heads, dropout=dropout, dim_feedforward=1000)
        self.encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=n_layers)
        
        self.mode = mode
        self.single_reduction = nn.Linear(n_heads * v_unit, 1)
        self.double_reduction = nn.Linear(2 * n_heads * v_unit, 1)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, data, direct):
        values = data[direct]['values'] #(batch,T,measure)
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)
        
        batch_size, t_dim, m_dim = values.shape
        
        values = values.unsqueeze(1) # [ batch, 1, T, measure]
        
        input_for_trans1 = self.conv1(values)
        input_for_trans2 = self.conv2(values) # ? [B, d, T, M]

        # ? [T, B, M, d], d_model = M * d
        input_for_trans1 =\
             input_for_trans1.permute(2, 0, 3, 1).reshape(t_dim, batch_size, -1) 
        # ? [M, B, T, d], d_model = T * d
        input_for_trans2 =\
             input_for_trans2.permute(3, 0, 2, 1).reshape(m_dim, batch_size, -1) 

        mask1 = generate_inf_mask(t_dim)
        output1 = self.encoder1(input_for_trans1,mask=mask1)
        
        mask2 = generate_inf_mask(m_dim)
        output2 = self.encoder2(input_for_trans2,mask=mask2)

        if self.mode == 'time':
            output1 = output1.reshape(t_dim, batch_size, m_dim, -1)
            output1 = self.single_reduction(output1)
            output1 = output1.squeeze(-1).permute(1,0,2)
            out_put = output1
        elif self.mode == 'measure':
            output2 = output2.reshape(m_dim, batch_size, t_dim, -1)
            output2 = self.single_reduction(output2)
            output2 = output2.squeeze(-1).permute(1,2,0)
            out_put = output2
        else:
            output3 = torch.cat((output1.reshape(t_dim, batch_size, m_dim, -1).transpose(0,1),
                        output2.reshape(m_dim, batch_size, t_dim, -1).permute(1,2,0,3)),3)
            output3 = self.double_reduction(output3)
            output3 = output3.squeeze(-1)
            out_put = output3
        
        x_loss = (masks*((out_put - values.squeeze(1)) ** 2)).reshape(batch_size,-1).sum(-1)/ \
                            masks.reshape(batch_size,-1).sum(-1)
        x_loss = torch.sum(x_loss) # ! 不用考虑is_train

        # import ipdb
        # ipdb.set_trace()

        return {'loss': x_loss, 'predictions': labels,\
                'imputations': out_put, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    
    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
