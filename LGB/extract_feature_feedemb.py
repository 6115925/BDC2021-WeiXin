import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from sklearn.decomposition import PCA
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

feed_info = feed_info[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']]
# 'manual_keyword_list', 'machine_keyword_list','manual_tag_list', 'machine_tag_list']]
embeddings = pd.read_csv('../data/wechat_algo_data1/feed_embeddings.csv')
feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds",]] = feed_info[["bgm_song_id", "bgm_singer_id", "videoplayseconds",]].fillna(0)

df = df.merge(feed_info, on='feedid', how='left')
col= list(df)

emb = embeddings['feed_embedding'].str.split(' ', expand=True)
emb = emb[emb.columns[:512]]
emb = emb.astype(float)
emb = emb.values
pca = PCA(n_components=64)
pca.fit(emb)
new_emb = pd.DataFrame(pca.transform(emb))

embeddings = pd.concat([embeddings, new_emb], axis=1)

embeddings = embeddings.drop(['feed_embedding'], axis=1)
embeddings.columns = embeddings.columns.map(lambda x: str(x))
df = df.merge(embeddings, how='left', on='feedid')



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

df=df.drop(col,axis=1)
df.to_pickle('extract_feature_feedemb.pkl')

# df.to_csv('extract_feature_feedemb.csv',index=None)