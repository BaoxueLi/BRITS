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

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

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
        N, T, C = query.shape

        values  = self.values(values)  # (N, T, embed_size)
        keys    = self.keys(keys)      # (N, T, embed_size)
        queries = self.queries(query)  # (N, T, embed_size)
        
        # Split the embedding into self.heads different pieces
        values = values.reshape(N, T, self.heads, self.head_dim)  #embed_size维拆成 heads×head_dim
        keys   = keys.reshape(N, T, self.heads, self.head_dim)
        query  = query.reshape(N, T, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("qthd,kthd->qkth", [queries, keys])   # 空间self-attention
        # queries shape: (N, T, heads, heads_dim),
        # keys shape: (N, T, heads, heads_dim)
        # energy: (N, N, T, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=1)  # 在K维做softmax，和为1
        # attention shape: (N, N, T, heads)

        out = torch.einsum("qkth,kthd->qthd", [attention, values]).reshape(
            N, T, self.heads * self.head_dim
        )        
        # attention shape: (N, N, T, heads)
        # values shape: (N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, T, embed_size)

        return out


class Model(nn.Module):
    def __init__(
        self, 
        adj,
        in_channels = 1, 
        embed_size = 64, 
        time_num = 100,
        num_layers = 3,
        T_dim = 12,
        output_T_dim = 3,  
        heads = 2,
        dropout = 0.1        
    ):
        super(Model, self).__init__()
        # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)

        # # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        # self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)  

        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        self.mha_t = MultiHeadAttention(d_model, num_heads, m_dim)
        self.mha_m = MultiHeadAttention(d_model, num_heads, time_dim)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.STransformer = STransformer(embed_size, heads, adj, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)

    
    def forward(self, value, key, query, t):
        # Add skip connection,run through normalization and finally dropout
        x1 = self.norm1(self.STransformer(value, key, query) + query)
        x2 = self.dropout( self.norm2(self.TTransformer(x1, x1, x1, t) + x1) )
        return x2

    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret


class Model(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight, label_weight,input_size):
        
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight
        self.label_weight = label_weight
        self.input_size = input_size

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.input_size * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size = self.input_size, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = self.input_size, output_size = self.input_size, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.input_size)
        self.feat_reg = FeatureRegression(self.input_size)

        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    def forward(self, data, direct):
        
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['labels'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss': x_loss * self.impute_weight + y_loss * self.label_weight, 'predictions': y_h,\
                'imputations': imputations, 'labels': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
