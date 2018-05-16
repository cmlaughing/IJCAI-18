import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.metrics import log_loss

import numpy as np
import pandas as pd

merge = pd.read_csv('../feat_file/basic_feat.csv')

#连接特征文件
feat_dir = '../feat_file/'
for feat_file in os.listdir(feat_dir):
    if not feat_file.startswith('.'):  #过滤掉隐藏文件
        temp = pd.read_csv(feat_dir+feat_file)
        merge = merge.merge(temp,how='left',on='row_id')
        print(feat_file+' loaded')

#去除不参与训练列
remove_col = ['item_category_list','predict_category_property','item_property_list','time','instance_id','user_id']
keep_columns = [item for item in merge.columns if item not in remove_col]

#使用第七天训练
train_data = merge[(merge.row_id<train_shape)&(merge.day==7)][keep_columns]
train_data.drop('row_id',axis=1,inplace=True)

#线下验证还是生成线上结果
online = False

if not online:
    #使用day 7 hour 11 作为线下验证集
    valid_x = train_data[(train_data.day==7) & (train_data.hour==11-i)]
    valid_x.drop(['day','hour'],axis=1,inplace=True)
    valid_y = valid_x.pop('is_trade')
    
    #使用day 7 hour 0~10 数据作为训练集
    drop_index = train_data[(train_data.day==7) & (train_data.hour>=11-i)].index
    train_x = train_data.drop(drop_index)
    train_x.drop(['day','hour'],axis=1,inplace=True)
    train_y = train_x.pop('is_trade')
    eval_set = [(train_x,train_y),(valid_x,valid_y)]
    
    #使用默认学习率0.1，方便快速线下验证loss
    clf = lgb.LGBMClassifier(max_depth=7,n_estimators=400,learning_rate=0.1)
    clf.fit(train_x,train_y,eval_set=eval_set)
    lgb_valid_pred = clf.predict_proba(valid_x)[:,1]
    metric_value = log_loss(valid_y,lgb_valid_pred)
    print('offline loss: ',metric_value)

del train_x,train_y ,eval_set
gc.collect()

if online:
    #调低学习率生成线上结果
    clf = lgb.LGBMClassifier(max_depth=7,n_estimators=3000,learning_rate=0.01)
    clf.fit(train_data.drop(['day','hour','is_trade'],axis=1),train_data['is_trade'])
    test_data = merge[merge.row_id>=train_shape][keep_columns].drop(['is_trade','row_id','hour','day'],axis=1)
    y_pred = clf.predict_proba(test_data)[:,1]
    result =  merge[merge.row_id>=train_shape]['instance_id'].to_frame().assign(predicted_score=y_pred)
    
    #读取test文件，按instance顺序merge结果生成提交文件
    instance_order = pd.read_csv('../raw_data/round2_ijcai_18_test_a_20180425.txt',sep=' ')[['instance_id']]
    submission = pd.merge(instance_order,result,how='left',on=['instance_id'])
    submission.to_csv('../result_file/submission.txt',index=False,sep=' ')
    print('submission finished')