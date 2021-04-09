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

def generate_mask(batch, atten_dim, heads):
    unit_mat = 1 - np.identity(atten_dim)
    unit_mat = torch.from_numpy(unit_mat).cuda()
    unit_mat = unit_mat.unsqueeze(0)
    unit_mat = unit_mat.unsqueeze(3)
    unit_mat = unit_mat.repeat(batch,1,1,heads)
    # import ipdb
    # ipdb.set_trace()
    return unit_mat.float()

def generate_mask_test(batch, atten_dim, heads):
    # ! alerts
    # ? test
    # TODO test
    unit_mat = np.identity(atten_dim)
    unit_mat = torch.from_numpy(unit_mat).cuda()
    unit_mat = unit_mat.unsqueeze(0)
    unit_mat = unit_mat.unsqueeze(3)
    unit_mat = unit_mat.repeat(batch,1,1,heads)
    # import ipdb
    # ipdb.set_trace()
    return unit_mat.float()

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=2*SEQ_LEN):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.unsqueeze(2)
        # pe.shape [1, SEQ_LEN, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape [batch, SEQ_LEN, measure, d_model]
        # import ipdb
        # ipdb.set_trace()
        return Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, t_dim, m_dim, T_unit, M_unit, V_unit, mode):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        if mode == 'spatial':
            self.extra_dim = t_dim
            self.head_dim = M_unit # 指attention时的维度，(batch, measure, heads * M_unit)
        elif mode == 'temporal':
            self.extra_dim = m_dim
            self.head_dim = T_unit
        # self.head_dim = embed_size * extra_dim // heads
        self.mode = mode
        self.V_unit = V_unit

        # assert (
        #     embed_size * extra_dim % heads == 0
        # ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(V_unit * heads, V_unit * heads, bias=False)
        self.keys = nn.Linear(self.extra_dim * V_unit * heads, self.heads * self.head_dim, bias=False)
        self.queries = nn.Linear(self.extra_dim * V_unit * heads, self.heads * self.head_dim, bias=False)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries):

        B, T, N, C = queries.shape

        if self.mode == 'spatial':
            # values = values.transpose(1,2) # (B, N, T, 1)
            keys = keys.transpose(1,2)
            queries = queries.transpose(1,2)
            atten_dim = N
        elif self.mode == 'temporal':
            atten_dim = T   # (B, T, N, 1)
            pass

        # 非attention的维度放在一起
        # values = values.reshape(B, atten_dim, -1)   # (B, T, N) or (B, N, T)
        keys   = keys.reshape(B, atten_dim, -1)  
        queries  = queries.reshape(B, atten_dim, -1)  

        values  = self.values(values)  # (B, T, N, heads * unit)
        # values_test = values           # ! test   
        keys    = self.keys(keys)      # (B, atten_dim, head * unit)
        queries = self.queries(queries)  # (B, atten_dim, head * unit)
      
        # Split the embedding into self.heads different pieces
        values = values.reshape(B, T, N, self.heads, self.V_unit)  #拆成 heads×head_dim
        # (B, T, N, heads, unit)
        keys   = keys.reshape(B, -1, self.heads, self.head_dim)
        queries  = queries.reshape(B, -1, self.heads, self.head_dim)
        # (B, atten_dim, heads, unit)
        
        energy = torch.einsum("bqhd,bkhd->bqkh", [queries, keys])
        # queries shape: (B, atten_dim, heads, heads_dim)
        # keys shape: (B, atten_dim, heads, heads_dim)
        # energy: (B, atten_dim, atten_dim, heads) 
        
        # import ipdb
        # ipdb.set_trace()

        if FLAG_IMPUTATION:
            cur_mask = generate_mask(B,atten_dim,self.heads)
            # cur_mask = generate_mask_test(B,atten_dim,self.heads) #! test
            energy = energy * cur_mask + (1-cur_mask) * (-1e9)
        
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=2)  # 在K维做softmax，和为1
        # (batch, query_len, key_len, heads) 

        if self.mode == 'spatial':
            # (B, T, N, heads, unit) -> (B, N, heads, T*unit)
            values=values.permute(0,2,3,1,4)
            values=values.reshape(B,N,self.heads,-1)
        elif self.mode == 'temporal':
            # (B, T, N, heads, unit) -> (B, T, heads, N*unit)
            values=values.permute(0,1,3,2,4)
            values=values.reshape(B,T,self.heads,-1)


        out = torch.einsum("bqkh,bkhd->bqhd", [attention, values])
        # attention shape: (B, atten_dim, atten_dim, heads)
        # values shape: (B, atten_dim, heads, extra_dim * heads_dim)
        # out after matrix multiply: (B, atten_dim, heads, extra_dim * heads_dim), then
        # we reshape. 
        out = out.reshape(B,atten_dim,self.heads,-1,self.V_unit)
        out = out.transpose(2,3)
        out=out.reshape(B,atten_dim,self.extra_dim,-1)
  

        if self.mode == 'spatial':
            out = out.transpose(1,2)
        # import ipdb
        # ipdb.set_trace()
        return out


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_size,   
        heads,
        T_unit,
        M_unit,
        V_unit,
        m_dim,
        t_dim,
        dropout = 0.1,
        forward_expansion = 4        
    ):
        super(EncoderLayer, self).__init__()

        # # 不能只用一个attention，不然就会共享linear层了
        self.mha_spatial = MultiHeadAttention(embed_size, heads, t_dim, m_dim,
                                            T_unit, M_unit, V_unit, 'spatial')
        self.mha_temporal = MultiHeadAttention(embed_size, heads, t_dim, m_dim, 
                                            T_unit, M_unit, V_unit, 'temporal')
        # 顺序 value,key,qurey

        self.feed_forward = nn.Sequential(
            nn.Linear(heads * V_unit, forward_expansion * heads * V_unit),
            nn.ReLU(),
            nn.Linear(forward_expansion * heads * V_unit, heads * V_unit),
        )

        self.reduction = nn.Linear(2 * heads * V_unit, heads * V_unit)

        self.norm1 = nn.LayerNorm(heads * V_unit)
        self.norm2 = nn.LayerNorm(heads * V_unit)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, atten_input):
        ##### T + S
        # ! pre-LN
        atten_input_pure = atten_input
        atten_input = self.norm1(atten_input)

        x0 = torch.cat((self.mha_spatial(atten_input, atten_input, atten_input),
                        self.mha_temporal(atten_input, atten_input, atten_input)),3)

        x1 = self.dropout1(
                self.reduction(x0)+
                atten_input_pure)
        ###### S only
        # x0 = self.mha_spatial(value, key_measure, atten_input)
        ###### T only
        # x0 = self.mha_temporal(value, key_time, atten_input)

        # x1 = self.dropout1(self.norm1(
        #         x0+
        #         atten_input))
        ########################
        # border 
        ########################
        x2 = self.dropout2(self.feed_forward(self.norm2(x1)) + x1)
        return x2

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
        v_unit = 3, #1->heads * v_unit
        n_heads=8, 
        n_layers=4
        ):
        super(Model, self).__init__()

        # # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(1, n_heads * v_unit, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(n_heads * v_unit, 1, 1)

        self.pos_emb = PositionalEncoding(
            d_model=embed_size,
            dropout=0)

        self.key_time_linear = nn.Linear(m_dim, n_heads * T_unit, bias=False)
        self.key_measure_linear = nn.Linear(t_dim, n_heads * M_unit, bias=False)
        
        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                embed_size=embed_size, heads=n_heads,
                T_unit=T_unit, M_unit=M_unit, V_unit=v_unit,
                m_dim=m_dim, t_dim=t_dim)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, data, direct):
        values = data[direct]['values'] #(batch,T,measure)
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)
        
        batch_size, t_dim, m_dim = values.shape
        
        enclayer_zero = values.unsqueeze(-1) # [ batch, T, measure, 1 ]
        values = values.unsqueeze(1)
        
        input_transformer = self.conv1(values)
        input_transformer = input_transformer.permute(0, 2, 3, 1) 
        # ! [ batch, T, measure, heads * v_unit = 24]

        #[batch, T, heads * T_unit]
        # K_time = self.key_time_linear(enclayer_zero.reshape(batch_size, t_dim, -1))
        #[batch, M, heads * M_unit]
        # K_measure = self.key_measure_linear(enclayer_zero.transpose(1,2).reshape(batch_size,m_dim,-1))

        # 暂时不要positional encodeing
        # pos_emb = self.pos_emb(input_transformer)

        enclayer_in = input_transformer

        for i in range(len(self.layers)):
            enclayer_in = self.layers[i](atten_input=enclayer_in)
        
        out_put = enclayer_in.permute(0, 3, 1, 2) #(B, T, M, d)->(B, d, T, M)
        out_put = self.conv2(out_put) #(B, d, T, M)->(B, 1, T, M)
        out_put = out_put.squeeze(1) # (batch, T, M)
        
        x_loss = (masks*((out_put - values.squeeze(1)) ** 2)).reshape(batch_size,-1).sum(-1)/ \
                            masks.reshape(batch_size,-1).sum(-1)
        # import ipdb
        # ipdb.set_trace()
        x_loss = torch.sum(x_loss * is_train.squeeze(1)) / (torch.sum(is_train) + 1e-5)


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
