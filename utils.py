import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random
import pandas as pd

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
            # print('This var is on GPU.')
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def coal_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(masks.shape[0]):
        if h == 0:
            deltas.append(np.ones(masks.shape[1]))
        else:
            deltas.append(np.ones(masks.shape[1]) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def coal_rec(values, masks, evals, eval_masks, dir_):
    deltas = coal_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values
    # forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    # rec['masks'] = masks.astype('int32').tolist()
    rec['masks'] = masks.astype('float64').tolist()
    # import ipdb
    # ipdb.set_trace()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    # rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['eval_masks'] = eval_masks.astype('float64').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec
