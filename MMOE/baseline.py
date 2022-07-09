import os
import pandas as pd
import numpy as np
import tensorflow as tf

from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from evaluation import evaluate_deepctr
from mmoe import MMOE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

from sklearn import metrics

# -*- coding: utf-8 -*-
##### 设置随机种子，使每次训练可重复
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 2021
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
# tf.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
# from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=24, inter_op_parallelism_threads=24)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

def fmean(df, f):
    df[f]=df[f].fillna(0)
    sentences = df[f].values.tolist()
    for i in range(len(sentences)):
        sentences[i] = [x for x in str(sentences[i]).split(';')]
    vec = []
    for seq in sentences:
        s=0
        for w in seq:
            s+=int(w)
        vec.append(s//len(seq))
    tmp = pd.DataFrame(vec,columns=[str(f)+'_mean'])
    return tmp





if __name__ == "__main__":
    submit = pd.read_csv('../data/wechat_algo_data1/test_a.csv')[['userid', 'feedid']]
    epochs = 2
    batch_size = 512
    embedding_dim = 64

    train_data = pd.read_csv('../data/wechat_algo_data1/user_action.csv')
    feed = pd.read_csv('../data/wechat_algo_data1/feed_info.csv')
    embedding = pd.read_pickle('extract_feature_embedding.pkl')

    feedemb = pd.read_pickle('extract_feature_feedemb.pkl')


    target = ["read_comment", "like", "click_avatar", "forward",]
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id'
                       ]
    dense_features = ['videoplayseconds',]+list(embedding)+list(feedemb)


    feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
    feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')
    feed["videoplayseconds"] = np.log(feed["videoplayseconds"] + 1.0)

    feed.drop(['description', 'ocr', 'asr', 'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list',
               'machine_tag_list', 'description_char', 'ocr_char', 'asr_char'], axis=1, inplace=True)

    train_data = train_data.merge(feed, how='left', on='feedid')


    test = pd.read_csv('../data/wechat_algo_data1/test_a.csv')
    test = test.merge(feed, how='left', on='feedid')

    data = pd.concat([train_data, test], axis=0, ignore_index=True)

    data = pd.concat([data, feedemb, embedding], axis=1)


    print('data.shape', data.shape)
    print('data.columns', data.columns.tolist())
    print('unique date_: ', data['date_'].unique())

    data[sparse_features] = data[sparse_features].fillna(0)
    data[dense_features] = data[dense_features].fillna(0)

    # # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                              for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    print(feature_names)
    # 3.generate input data for model
    new_data, test = data.iloc[:train_data.shape[0]].reset_index(drop=True), data.iloc[train_data.shape[0]:].reset_index(drop=True)

    train = new_data[(new_data['date_'] < 14) ]
    val = new_data[new_data['date_'] == 14]  # 第14天样本作为验证集


    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}

    userid_list = val['userid'].astype(str).tolist()
    test_model_input = {name: test[name] for name in feature_names}

    train_labels = [train[y].values for y in target]
    val_labels = [val[y].values for y in target]

    # 4.Define Model,train,predict and evaluate
    train_model = MMOE(dnn_feature_columns, num_tasks=4,num_experts=4, expert_dim=8, dnn_hidden_units=(512,512),seed=2021,
                       tasks=['binary', 'binary', 'binary', 'binary'])
    train_model.compile("adagrad", loss='binary_crossentropy')
    # print(train_model.summary())
    for epoch in range(epochs):
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1)
        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)

    # t1 = time()
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 10)


    # t2 = time()
    # print('4个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
    # ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
    # print('4个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)

    # 5.生成提交文件
    for i, action in enumerate(target):
        submit[action] = pred_ans[i]
    submit[['userid', 'feedid'] + target].to_csv('baseline.csv', index=None, float_format='%.6f')
    print('to_csv ok')
