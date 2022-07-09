import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from gensim.models import Word2Vec
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

feed_info = feed_info[['feedid', 'authorid', 'bgm_song_id','bgm_singer_id','videoplayseconds','manual_tag_list']]
# 'manual_keyword_list', 'machine_keyword_list','manual_tag_list', 'machine_tag_list']]
embeddings = pd.read_csv('../data/wechat_algo_data1/feed_embeddings.csv')

feed_info[["bgm_song_id", "bgm_singer_id", ]] = feed_info[["bgm_song_id", "bgm_singer_id"]].fillna(0)

fn = feed_info ['manual_tag_list'].str.split(';', expand=True)
fn = fn.astype(float)
fn = fn.fillna(0)
fn['tag'] = fn[fn.columns[0]]
feed_info = pd.concat([feed_info, fn['tag']], axis=1)
feed_info = feed_info[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id','bgm_singer_id','tag']]
df = df.merge(feed_info, on='feedid', how='left')

col= list(df)
print(col)
sort_df=df
def emb(df, f1, f2,emb_size):
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, vector_size=emb_size, window=5,
                     min_count=5, sg=0, hs=1,workers=1, seed=2021)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    df_emb = pd.DataFrame(emb_matrix)
    df_emb.columns = ['{}_{}_emb_{}'.format(
        f1, f2, i) for i in range(emb_size)]
    tmp = pd.concat([tmp, df_emb], axis=1)
    del model.wv, emb_matrix, sentences
    return tmp

for f1, f2 ,f3 in [
                ['userid', 'feedid',64],
                ['userid', 'authorid',64],
                # ['userid', 'bgm_song_id',32],
                # ['userid', 'bgm_singer_id',32],
                ['feedid', 'userid', 64],
                ['authorid', 'userid', 64],
                # [ 'bgm_song_id','userid',16],
                # ['bgm_singer_id','userid', 16],

                ]:
    df = df.merge(emb(sort_df, f1, f2,f3), on=f1, how='left')
del sort_df
gc.collect()
#
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
# 内存够用的不需要做这一步
df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])
print(df[:4])
df=df.drop(col,axis=1)
df.to_pickle('extract_feature_embedding.pkl')
