import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
np.random.seed(2021)

y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15
## 读取训练集
train = pd.read_csv('../data/wechat_algo_data1/user_action.csv')
print(train.shape)
for y in y_list:
    print(y, train[y].mean())

## 读取测试集
test = pd.read_csv('../data/wechat_algo_data1/test_a.csv')
test['date_'] = max_day
print(test.shape)

## 合并处理
df = pd.concat([train, test], axis=0, ignore_index=True)
print(df.head(3))
## 读取视频信息表
feed_info = pd.read_csv('../data/wechat_algo_data1/feed_info.csv')
feed_info = feed_info[['feedid', 'authorid', 'videoplayseconds','manual_keyword_list', 'machine_keyword_list','manual_tag_list']]
# 'manual_keyword_list', 'machine_keyword_list','manual_tag_list', 'machine_tag_list']]
embeddings = pd.read_csv('../data/wechat_algo_data1/feed_embeddings.csv')

fn = feed_info ['manual_tag_list'].str.split(';', expand=True)
fn = fn.astype(float)
fn = fn.fillna(0)
fn['tag'] = fn[fn.columns[0]]
feed_info = pd.concat([feed_info, fn['tag']], axis=1)
feed_info = feed_info[['feedid', 'authorid', 'videoplayseconds','tag']]

df = df.merge(feed_info, on='feedid', how='left')


## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df['videoplayseconds'] *= 1000
## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
play_cols = ['is_finish', 'play_times','play', 'stay']
df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')

# 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 10
for stat_cols in tqdm([
    ['userid'],
    ['feedid'],
    ['authorid'],
    ['tag'],
    ['userid', 'authorid'],
    ['userid', 'tag'],
]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1
        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
        g = tmp.groupby(stat_cols)
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]
        for x in play_cols[1:]:
            for stat in ['max', 'mean']:
                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))
        for y in y_list[:4]:
            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])
        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del g, tmp
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')


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
play_cols = ['is_finish', 'play_times','play', 'stay']
print(df.shape)
# 内存够用的不需要做这一步
df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])
df.to_pickle('extract_feature_history_1.pkl')
