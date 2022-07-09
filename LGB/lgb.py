import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc
import time
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import random
pd.set_option('display.max_columns', None)

np.random.seed(2021)

def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

## 从官方baseline里面抽出来的评测函数
def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)
    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag
    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc

y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

##

cross = pd.read_pickle('extract_feature_cross.pkl')
history = pd.read_pickle('extract_feature_history.pkl')
embedding = pd.read_pickle('extract_feature_embedding.pkl')
# embedding_10 = pd.read_pickle('extract_feature_embedding_10.pkl')
feedemb = pd.read_pickle('extract_feature_feedemb.pkl')
# deepwalk = pd.read_pickle('extract_feature_deepwalk.pkl')

df = pd.concat([history,cross,feedemb,embedding],axis=1)
print('read finish')


print(df[:4],df[-4:])

play_cols = ['is_finish', 'play_times','play', 'stay']
# # 内存够用的不需要做这一步
# df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])
# print(df[:4])
train = df[~df['read_comment'].isna()].reset_index(drop=True)
test = df[df['read_comment'].isna()].reset_index(drop=True)
cols = [f for f in df.columns if f not in ['date_']+ play_cols + y_list]
df = df.fillna(0)


del df
gc.collect()

# + play_cols + y_list]
print(train[cols].shape)
print(cols)

trn_x = train[train['date_'] < 14].reset_index(drop=True)
val_x = train[train['date_'] == 14].reset_index(drop=True)


##################### 线下验证 #####################
uauc_list = []
r_list = []
for y in y_list[:4]:
    print('=========', y, '=========')
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = cols
    t = time.time()
    clf = LGBMClassifier(
        learning_rate=0.01,
        n_estimators=5000,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2021,
        metric='None',
    )
    clf.fit(
        trn_x[cols], trn_x[y],
        eval_set=[(val_x[cols], val_x[y])],
        eval_metric='auc',
        early_stopping_rounds=100,
        verbose=50
    )
    fold_importance_df[f'imp'] = clf.feature_importances_
    val_x[y + '_score'] = clf.predict_proba(val_x[cols])[:, 1]
    val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])
    uauc_list.append(val_uauc)
    print(val_uauc)
    fold_importance_df.sort_values(by='imp', ascending=False, inplace=True)
    print(fold_importance_df[['Feature', 'imp']])
    r_list.append(clf.best_iteration_)
    print('runtime: {}\n'.format(time.time() - t))
weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
print(uauc_list)
print(weighted_uauc)
del trn_x ,val_x
gc.collect()
##################### 全量训练 #####################
r_dict = dict(zip(y_list[:4], r_list))
for y in y_list[:4]:
    print('=========', y, '=========')
    t = time.time()
    clf = LGBMClassifier(
        learning_rate=0.01,
        n_estimators=r_dict[y],
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2021,
        metric='auc',
    )
    clf.fit(
        train[cols], train[y],
        eval_set=[(train[cols], train[y])],
        early_stopping_rounds=r_dict[y],
        eval_metric='auc',
        verbose=100
    )
    test[y] = clf.predict_proba(test[cols])[:, 1]
    print('runtime: {}\n'.format(time.time() - t))
test[['userid', 'feedid'] + y_list[:4]].to_csv('sub_%.6f_%.6f_%.6f_%.6f_%.6f.csv' % (weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3]),
    index=False
)

