import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time, os
import missingno as msno
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import MinMaxScaler
from scipy import stats
import warnings
from utils import *
from sklearn.model_selection import train_test_split
import argparse

options_parser = argparse.ArgumentParser()
options_parser.add_argument('--miss_rate', type=float, default=0.2,
                    help='Missing rate')
options_parser.add_argument('--win_len', type=float, default=100)
options_parser.add_argument('--burst_min', type=float, default=10)
options_parser.add_argument('--burst_max', type=float, default=50)
options_parser.add_argument('--re_mask', type=int, default=0)
options_parser.add_argument('--seed', type=int, default=1)
options_parser.add_argument('--split_ratio', type=float, default=0.7)
options = options_parser.parse_args()

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

data_A=pd.read_csv("E://newJupyter//MissImputation//clouddata//8机磨煤机A正常数据-20190910-20191103.csv")
data_B=pd.read_csv("E://newJupyter//MissImputation//clouddata//8机磨煤机B正常数据-20191104-20191117.csv")
data_D=pd.read_csv("E://newJupyter//MissImputation//clouddata//8机磨煤机D正常数据-20190601-20190804.csv")
data_E=pd.read_csv("E://newJupyter//MissImputation//clouddata//8机磨煤机E正常数据-20190414-20190428.csv")
data_F=pd.read_csv("E://newJupyter//MissImputation//clouddata//8机磨煤机F正常数据-20190801-20190915.csv")

# 去掉最后一行
data_A.drop([len(data_A)-1],inplace=True)
data_B.drop([len(data_B)-1],inplace=True)
data_D.drop([len(data_D)-1],inplace=True)
data_E.drop([len(data_E)-1],inplace=True)
data_F.drop([len(data_F)-1],inplace=True)
# drop the "Time" column
data_A.drop('Time', axis=1, inplace=True)
data_B.drop('Time', axis=1, inplace=True)
data_D.drop('Time', axis=1, inplace=True)
data_E.drop('Time', axis=1, inplace=True)
data_F.drop('Time', axis=1, inplace=True)
# 把报错改为NULL
data_A=data_A[~data_A.isin(["Equip Fail"])]
data_B=data_B[~data_B.isin(["Equip Fail"])]
data_D=data_D[~data_D.isin(["Equip Fail"])]
data_E=data_E[~data_E.isin(["Equip Fail"])]
data_F=data_F[~data_F.isin(["Equip Fail"])]

data_A=data_A[~data_A.isin(["Bad"])]
data_B=data_B[~data_B.isin(["Bad"])]
data_D=data_D[~data_D.isin(["Bad"])]
data_E=data_E[~data_E.isin(["Bad"])]
data_F=data_F[~data_F.isin(["Bad"])]

# 停机肯定是要去掉的，但不是现在
# data_D=data_D[~(data_D[data_D.columns[0]]<20)] # 去掉停机
# data_F=data_F[~(data_F[data_F.columns[0]]<20)]
# data_B=data_B[~(data_B[data_B.columns[0]]<20)] # 去掉停机
# data_E=data_E[~(data_E[data_E.columns[0]]<20)]
# #去掉某些行后需要对index重新索引
# data_B=data_B.reset_index(drop=True)
# data_E=data_E.reset_index(drop=True)
# data_D=data_D.reset_index(drop=True)
# data_F=data_F.reset_index(drop=True)

# 调换B的属性顺序,注意，这里把time维度去掉了，所以index都减1
B_col=data_B.columns.tolist()
temp_col=B_col.pop(14)
B_col.insert(12,temp_col)
data_B=data_B[B_col]

B_col=data_B.columns.tolist()
temp_col=B_col.pop(15)
B_col.insert(13,temp_col)
data_B=data_B[B_col]

B_col=data_B.columns.tolist()
temp_col=B_col.pop(16)
B_col.insert(14,temp_col)
data_B=data_B[B_col]
########################
# 把BDEF的放在一起看看 #
########################
# name_list=['B','D','E','F']
# data_list=[data_B.copy(),data_D.copy(),data_E.copy(),data_F.copy()]
name_list=['D','B','E','F']
data_list=[data_D.copy(),data_B.copy(),data_E.copy(),data_F.copy()]
# print(len(data_D),len(data_F))

# 生成一份去掉停机数据的数据（这是考虑到停机数据会对标准化有影响），求取平均值和方差（或最大值最小值）
# 标准化

setup_seed(options.seed)

for j in range(len(data_list)):
    df=data_list[j]
    print('coalmill No.{}'.format(j))
    # 统计缺失情况
    empty_column = []
    for e, c in enumerate(df.columns):
        if sum(pd.isnull(df[c]))!=0:
            empty_column.append(c)
            print("feature_no:%d \t feature_name:%s \t null_num:%d \t null_rate: %.2f%%"\
                        % (e, c.split('-')[1] , sum(pd.isnull(df[c])), 
                        100*sum(pd.isnull(df[c]))/len(df[c])))
    if empty_column == []:
        print('No missing.')
    # 非停机状况下的标准化
    data_work=data_list[j].copy()
    data_work=data_work[~(data_work[data_work.columns[0]]<20)]
    scaler = StandardScaler()
    scaler.fit(data_work)
    # sns.distplot(data_work[data_work.columns[0]],color='k')
    # plt.show()

    # 自带的mask记为-1
    data_noisy=np.array(df.copy()).astype('float')
    data_ground=np.array(df.copy()).astype('float')
    masks = (~np.isnan(data_ground)).astype('int')
    masks[masks==0] = -1
    idx_non_zero = np.where(masks == 1)
    n_non_zeros = idx_non_zero[0].shape[0]
    miss_size = n_non_zeros * options.miss_rate
    # miss_size = 2e4 # 测试专用
    print('All point:',df.shape[0]*df.shape[1])
    print('Miss size:',miss_size)
    print('Masking data......')
    if os.path.exists('./coalmill-mask/mask_{}.npy'.format(name_list[j])) or (options.re_mask!=0):
        masks=np.load('./coalmill-mask/mask_{}.npy'.format(name_list[j]))
        print('Load mask success.')
        data_noisy[masks==0] = np.nan
        # import ipdb
        # ipdb.set_trace()
        # # 再保存一下data_ground给mean、knn作为baseline
        #####################################
        # 不生成时要注释掉，因为代码段会改变data
        #####################################
        # ############
        # # 去掉停机
        # ############
        # masks = masks[data_ground[:,0]>20]
        # data_ground = data_ground[data_ground[:,0]>20]
        # #############
        # # 去掉自然缺失
        # #############
        # data_ground = np.delete(data_ground, np.where(masks==-1)[0], axis=0)
        # masks = np.delete(masks, np.where(masks==-1)[0], axis=0)
        # # 标准化
        # data_after_scale = scaler.transform(data_ground)

        # overall_num = masks.shape[0]
        # train_data = data_after_scale[:int(overall_num*(options.split_ratio)),:]
        # train_mask = masks[:int(overall_num*(options.split_ratio)),:]
        # test_data = data_after_scale[int(overall_num*(options.split_ratio)):,:]
        # test_mask = masks[int(overall_num*(options.split_ratio)):,:]

        # np.save('./coalmill-data-simple/data_{}_feature-{}.npy'.format(name_list[j],'train'),train_data)
        # np.save('./coalmill-data-simple/data_{}_mask-{}.npy'.format(name_list[j],'train'),train_mask)
        # np.save('./coalmill-data-simple/data_{}_feature-{}.npy'.format(name_list[j],'test'),test_data)
        # np.save('./coalmill-data-simple/data_{}_mask-{}.npy'.format(name_list[j],'test'),test_mask)
    else: 
        # 生成mask，比较耗时
        st_time = time.time()
        while np.where(masks == 0)[0].shape[0] < miss_size:
            coordi_x = np.random.randint(0, masks.shape[1])
            coordi_y = np.random.randint(0, masks.shape[0])
            burst_len = np.random.randint(options.burst_min,options.burst_max)
            judge_res=(masks[coordi_y:coordi_y+burst_len,coordi_x]==[1])
            if judge_res.all() == True: #如果都是1，即都没有缺失或人工缺失
                data_noisy[coordi_y:coordi_y+burst_len,coordi_x] = np.nan
                masks[coordi_y:coordi_y+burst_len,coordi_x] = 0
        np.save('./coalmill-mask/mask_{}.npy'.format(name_list[j]),masks)
        print('Save mask success.')
        en_time = time.time()
        print('Successful masking, time cosumed: {:.2f}s'.format(en_time-st_time))

    msno.matrix(pd.DataFrame(data_noisy[:5000,:]), labels=False)
    plt.savefig('visual/matrix_{}.pdf'.format(name_list[j]),dpi=300,bbox_inches='tight')
    msno.matrix(pd.DataFrame(data_ground[:5000,:]), labels=False)
    plt.savefig('visual/matrix_{}_origin.pdf'.format(name_list[j]),dpi=300,bbox_inches='tight')
    # 储存最终结果
    list_final=[]
    data_noisy=scaler.transform(data_noisy)
    data_ground=scaler.transform(data_ground)
    #### 减少数据量
    small_or_not = 1
    if small_or_not == 1:
        # 为了测试专用，加快加载速度
        data_noisy=data_noisy.copy()[:200,:]
        data_ground=data_ground.copy()[:200,:]
        masks=masks.copy()[:200,:]
        pass
    else:
        data_noisy=data_noisy.copy()[:5000,:]
        data_ground=data_ground.copy()[:5000,:]
        masks=masks.copy()[:5000,:]
    for k in range(masks.shape[0]-options.win_len):
        values = data_noisy[k:k+options.win_len,:].copy()
        value_mask = (~np.isnan(values)).astype('int')
        evals = data_ground[k:k+options.win_len,:].copy()
        eval_mask = masks[k:k+options.win_len,:].copy() # if not copy, masks will be changed
        eval_mask[eval_mask==1]=-1
        eval_mask[eval_mask==0]=1
        eval_mask[eval_mask==-1]=0
        assert (
            np.where(masks[k:k+options.win_len,:] == -1)[0].shape[0] + 
            np.where(eval_mask == 1)[0].shape[0] == 
            np.where(np.isnan(values))[0].shape[0]
        ), "Total missing should be equal to natural missing and artificial missing"
        # forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values
        rec={'label':0}
        rec['forward']=coal_rec(values,value_mask,evals,eval_mask,dir_='forward')
        rec['backward']=coal_rec(values[::-1],value_mask[::-1],evals[::-1],eval_mask[::-1],dir_='backward')
        list_final.append(rec)
        # import ipdb
        # ipdb.set_trace()
    if small_or_not:
        np.save('coalmill-data-TS/data_{}_small.npy'.format(name_list[j]),np.array(list_final))
    else:
        np.save('coalmill-data-TS/data_{}.npy'.format(name_list[j]),np.array(list_final))
import ipdb
ipdb.set_trace()

#TODO 还没去掉停机、还没去掉自然缺失、还没标准化、生成训练和测试(dict for rnn)(npy for simple，提前分好)

