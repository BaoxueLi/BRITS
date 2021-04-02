# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.filterwarnings('ignore')
import json
import fancyimpute
import numpy as np
import pandas as pd

from missingpy import MissForest
X = []
Y = []
Z = []

# for ctx in open('json/json'):
#     z = json.loads(ctx)['label']
#     ctx = json.loads(ctx)['forward']
#     x = np.asarray(ctx['values'])
#     y = np.asarray(ctx['evals'])


#     x_mask = np.asarray(ctx['masks']).astype(np.bool)
#     y_mask = np.asarray(ctx['eval_masks']).astype(np.bool)

#     x[~x_mask] = np.nan

#     y[(~x_mask) & (~y_mask)] = np.nan

#     X.append(x)
#     Y.append(y)
#     Z.append(int(z))

content = np.load('coalmill-data-TS/data_D_small.npy',allow_pickle=True).tolist()
len_content = len(content)

for j in range(len_content):
    z = content[j]['label']
    # import ipdb
    # ipdb.set_trace()
    ctx = content[j]['forward']
    x = np.asarray(ctx['values'])
    y = np.asarray(ctx['evals'])


    x_mask = np.asarray(ctx['masks']).astype(bool)
    y_mask = np.asarray(ctx['eval_masks']).astype(bool)

    x[~x_mask] = np.nan

    y[(~x_mask) & (~y_mask)] = np.nan

    X.append(x)
    Y.append(y)
    Z.append(int(z))

def get_loss(X, X_pred, Y):
    # find ones in Y but not in X (ground truth)
    mask = np.isnan(X) ^ np.isnan(Y)

    X_pred = np.nan_to_num(X_pred)
    pred = X_pred[mask]
    label = Y[mask]

    mae = np.abs(pred - label).sum() / (1e-5 + np.sum(mask))
    mre = np.abs(pred - label).sum() / (1e-5 + np.sum(np.abs(label)))
    rmse = np.sqrt(np.mean((pred-label)**2))
    return {'mae': mae, 'mre': mre,'rmse': rmse}

# Algo1: Mean imputation

X_mean = []

print(len(X))

for x, y in zip(X, Y):
    X_mean.append(fancyimpute.SimpleFill().fit_transform(x))

X_c = np.concatenate(X, axis=0).reshape(-1, 100, 36)
Y_c = np.concatenate(Y, axis=0).reshape(-1, 100, 36)
Z_c = np.array(Z)
X_mean = np.concatenate(X_mean, axis=0).reshape(-1, 100, 36)

print('Mean imputation:')
print(get_loss(X_c, X_mean, Y_c))

# save mean inputation results
print(X_c.shape, Y_c.shape, Z_c.shape)
# raw_input()
np.save('./result/mean_data.npy', X_mean)
np.save('./result/mean_label.npy', Z_c)


# Algo2: KNN imputation

X_knn = []

for x, y in zip(X, Y):
    X_knn.append(fancyimpute.KNN(k=10, verbose=False).fit_transform(x))


X_c = np.concatenate(X, axis=0)
Y_c = np.concatenate(Y, axis=0)
X_knn = np.concatenate(X_knn, axis=0)

print('KNN imputation')
print(get_loss(X_c, X_knn, Y_c))

# import ipdb
# ipdb.set_trace()

# ### Matrix Factorization
# since MF is extremely slow, we evaluate the imputation result every 100 iterations

X_mf = []

for i, (x, y) in enumerate(zip(X, Y)):
    X_mf.append(fancyimpute.MatrixFactorization(loss='mae', verbose=False).fit_transform(x))
    if i % 10 == 0:
        print(i)

X_c = np.concatenate(X, axis=0)
Y_c = np.concatenate(Y, axis=0)
X_mf = np.concatenate(X_mf, axis=0)

print('MF imputation')
print(get_loss(X_c, X_mf, Y_c))

# MICE imputation
# Since MICE can not handle the singular matrix, we do it in a batch style

X_mice = []

for x, y in zip(X, Y):
    # print('mice...')
    X_mice.append(MissForest().fit_transform(x))
    # import ipdb
    # ipdb.set_trace()
X_mice = np.concatenate(X_mice, axis=0)
X_c = np.concatenate(X, axis=0)
Y_c = np.concatenate(Y, axis=0)

print('MICE imputation')
print(get_loss(X_c, X_mice, Y_c))
