import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from ipdb import set_trace
from sklearn import metrics

SEQ_LEN = 100

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
    def __init__(self, embed_size, heads, mode):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.mode = mode
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.values = nn.Linear(self.embed_size, self.embed_size)
        self.keys = nn.Linear(self.embed_size, self.embed_size)
        self.queries = nn.Linear(self.embed_size, self.embed_size)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):

        B, T, N, C = query.shape

        values  = self.values(values)  # (B, T, N, embed_size)
        keys    = self.keys(keys)      # (B, T, N, embed_size)
        queries = self.queries(query)  # (B, T, N, embed_size)
        
        # Split the embedding into self.heads different pieces
        values = values.reshape(B, T, N, self.heads, self.head_dim)  #embed_size维拆成 heads×head_dim
        keys   = keys.reshape(B, T, N, self.heads, self.head_dim)
        queries  = queries.reshape(B, T, N, self.heads, self.head_dim)
        # (B, T, N, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        if self.mode == 'spatial':
            energy = torch.einsum("btqhd,btkhd->btqkh", [queries, keys])   # 空间self-attention
            # queries shape: (B, T, N, heads, heads_dim) 
            # keys shape: (B, T, N, heads, heads_dim) 
            # energy: (B, T, N, N, heads)
        elif self.mode == 'temporal':
            energy = torch.einsum("bqnhd,bknhd->bqknh", [queries, keys])   # 时间self-attention
            # queries shape: (B, T, N, heads, heads_dim) 
            # keys shape: (B, T, N, heads, heads_dim) 
            # energy: (B, T, T, N, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        if self.mode == 'spatial':
            cur_dim = 3
        elif self.mode == 'temporal':
            cur_dim = 2
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=cur_dim)  # 在K维做softmax，和为1

        # attention shape: (batch, T, query_len, key_len, heads) 
        # or (batch, query_len, key_len, N, heads)
        
        if self.mode == 'spatial':
            out = torch.einsum("btqkh,btkhd->btqhd", [attention, values]).reshape(
            B, T, N, self.heads * self.head_dim
            ) 
            # attention shape: (B, T, N, N, heads)
            # values shape: (B, T, N, heads, heads_dim) 
            # out after matrix multiply: (B, T, N, heads, heads_dim) , then
            # we reshape and flatten the last two dimensions.
      
        elif self.mode == 'temporal':
            out = torch.einsum("bqknh,bknhd->bqnhd", [attention, values]).reshape(
            B, T, N, self.heads * self.head_dim
            )
            # attention shape: (B, T, T, N, heads)
            # values shape: (B, T, N, heads, heads_dim)
            # out after matrix multiply: (B, T, N, heads, heads_dim), then
            # we reshape and flatten the last two dimensions. 

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (B, T, N, embed_size)

        return out


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_size,   
        heads,
        dropout = 0.1,
        forward_expansion = 4        
    ):
        super(EncoderLayer, self).__init__()

        # # 不能只用一个attention，不然就会共享linear层了
        self.mha_spatial = MultiHeadAttention(embed_size, heads, 'spatial')
        self.mha_temporal = MultiHeadAttention(embed_size, heads, 'temporal')
        # 顺序 value,key,qurey

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, value, value_plus_pos):
        # Add skip connection,run through normalization and finally dropout
        x1 = self.dropout1(self.norm1(
                self.mha_spatial(value, value, value) +\
                self.mha_temporal(value_plus_pos, value_plus_pos, value_plus_pos) + value))
        x2 = self.dropout2(self.norm2(self.feed_forward(x1) + x1))
        return x2

class Model(nn.Module):

    def __init__(self,
        rnn_hid_size, 
        impute_weight, 
        label_weight,
        input_size,
        embed_size=32,
        n_heads=8, 
        n_layers=4
        ):
        super(Model, self).__init__()

        # # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(1, embed_size, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(embed_size, 1, 1)

        self.pos_emb = PositionalEncoding(
            d_model=embed_size,
            dropout=0)

        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                embed_size=embed_size,heads=n_heads)
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
        
        batch_size = values.shape[0]
        values = values.unsqueeze(1)
        input_transformer = self.conv1(values)
        input_transformer = input_transformer.permute(0, 2, 3, 1) 
        # [ batch, T, measure, emb ]

        pos_emb = self.pos_emb(input_transformer)

        for i in range(len(self.layers)):
            if i == 0:
                input_transformer = self.layers[i](value=input_transformer,
                                            value_plus_pos=input_transformer+pos_emb)
            else:
                input_transformer = self.layers[i](value=input_transformer,
                                            value_plus_pos=input_transformer)
        
        out_put = input_transformer.permute(0, 3, 1, 2)
        out_put = self.conv2(out_put)
        out_put = out_put.squeeze(1) # (batch, T, measure)
        
        x_loss = (masks*((out_put - values.squeeze(1)) ** 2)).reshape(batch_size,-1).mean(-1)
        x_loss = x_loss * is_train
        import ipdb
        ipdb.set_trace()

        return {'loss': x_loss, 'predictions': 0,\
                'imputations': out_put, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    
    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
