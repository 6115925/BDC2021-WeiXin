import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
import gc
np.random.seed(2021)
import random
random.seed(2021)
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


sort_df=df


def deepwalk(df, f1, f2):
    L = 16
    # Deepwalk算法，
    print("deepwalk:", f1, f2)
    # 构建图
    dic = {}
    for item in df[[f1, f2]].values:
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_' + str(int(item[1]))].add('user_' + str(int(item[0])))
        except:
            dic['item_' + str(int(item[1]))] = set(['user_' + str(int(item[0]))])
        try:
            dic['user_' + str(int(item[0]))].add('item_' + str(int(item[1])))
        except:
            dic['user_' + str(int(item[0]))] = set(['item_' + str(int(item[1]))])
    dic_cont = {}
    for key in dic:
        dic[key] = list(dic[key])
        dic_cont[key] = len(dic[key])
    print("creating")
    # 构建路径
    path_length = 10
    sentences = []
    length = []
    for key in dic:
        sentence = [key]
        while len(sentence) != path_length:
            key = dic[sentence[-1]][random.randint(0, dic_cont[sentence[-1]] - 1)]
            if len(sentence) >= 2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences) % 100000 == 0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    # 训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, vector_size=L, window=5, min_count=1, sg=1, workers=1, seed=2021)
    print('outputing...')
    # 输出
    values = set(df[f1].values)
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model.wv['user_' + str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df1 = pd.DataFrame(w2v)
    names = [f1]
    for i in range(L):
        names.append(f1 + '_' + f2 + '_' + names[0] + '_deepwalk_embedding_' + str(L) + '_' + str(i))
    print(names)
    out_df1.columns = names
    print(out_df1.head())

    ########################
    values = set(df[f2].values)
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model.wv['item_' + str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df2 = pd.DataFrame(w2v)
    names = [f2]
    for i in range(L):
        names.append(f1 + '_' + f2 + '_' + names[0] + '_deepwalk_emb_' + str(L) + '_' + str(i))
    out_df2.columns = names
    print(out_df2.head())
    return (out_df1, out_df2)

emb_cols = [
    ['userid', 'feedid'],
    ['userid', 'authorid'],
]
for f1, f2 in emb_cols:
    out_df1, out_df2 = deepwalk(sort_df, f1, f2)
    df = df.merge(out_df1, on=f1, how='left')
    df = df.merge(out_df2, on=f2, how='left')

print(df)

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
df.to_pickle('extract_feature_deepwalk.pkl')
