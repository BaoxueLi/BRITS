import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
import time
import utils
import models
import argparse
import data_loader
import pandas as pd
# import ujson as json
from utils import setup_seed
from sklearn import metrics
import math
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str)
parser.add_argument('--domain', type=str, default='D')
parser.add_argument('--hid_size', type=int)
parser.add_argument('--input_size', type=int)
parser.add_argument('--impute_weight', type=float)
parser.add_argument('--label_weight', type=float)
parser.add_argument('--small_data', type=str,default='small')
parser.add_argument('--warm_up_epochs', type=int, default=10)
args = parser.parse_args()

best_res = {'epoch':0,'MAE':1e6,'MRE':1e6,'RMSE':1e6}

def train(model):
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (math.cos((epoch - args.warm_up_epochs) /(args.epochs - args.warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    data_iter = data_loader.get_loader(batch_size=args.batch_size,small=args.small_data,domain=args.domain)

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()

            print ('\r Progress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)),)

        evaluate(model, data_iter, epoch)
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()

def evaluate(model, val_iter, epoch):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    save_impute = []
    save_label = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())
        save_label.append(ret['labels'].data.cpu().numpy())

        pred = ret['predictions'].data.cpu().numpy()
        # import ipdb
        # ipdb.set_trace()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        # import ipdb
        # ipdb.set_trace()
        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    if np.abs(evals - imputations).mean() < best_res['MAE']:
        best_res['epoch'] = epoch
        best_res['MAE'] = np.abs(evals - imputations).mean()
        best_res['MRE'] = np.abs(evals - imputations).sum() / np.abs(evals).sum()
        best_res['RMSE'] = np.sqrt(np.mean((evals-imputations)**2))
    
    print('Epoch: {}, best epoch: {}'.format(epoch,best_res['epoch']))
    print ('MAE:', np.abs(evals - imputations).mean(),'best MAE:',best_res['MAE'])
    print ('MRE:', np.abs(evals - imputations).sum() / np.abs(evals).sum(),'best MRE:',best_res['MRE'])
    print ('RMSE', np.sqrt(np.mean((evals-imputations)**2)),'best RMSE',best_res['RMSE'])
    # import ipdb
    # ipdb.set_trace()

def run():
    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight, args.label_weight,args.input_size)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda(args.gpu)
        print('Model is on GPU.')

    train(model)

# python main.py --model brits --small_data small --input_size 36 --epochs 1000 --batch_size 64 --impute_weight 0.3 --label_weight 0 --hid_size 108
# python main.py --model CDSA --small_data small --domain F --input_size 36 --epochs 1000 --batch_size 64 --impute_weight 0.3 --label_weight 0 --hid_size 108
# python main.py --model MIDAI --small_data small --domain F --input_size 36 --epochs 1000 --batch_size 64 --impute_weight 0.3 --label_weight 0 --hid_size 108
if __name__ == '__main__':
    setup_seed(1)
    run()
    np.save('best_res/{}_{}_{}.npy'.format(args.model,args.domain,args.small_data),np.array(best_res))
    # ? np.load('best_res/MIDAI_F_medium.npy',allow_pickle=True)
