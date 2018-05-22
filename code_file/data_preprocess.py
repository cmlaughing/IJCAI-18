import numpy as np
import pandas as pd
import gc

train = pd.read_csv('../raw_data/round2_train.txt',sep=' ')
test_a = pd.read_csv('../raw_data/round2_ijcai_18_test_a_20180425.txt', sep=' ')
test_b = pd.read_csv('../raw_data/round2_ijcai_18_test_b_20180510.txt',sep=' ')

print('train data shape: ',train.shape)
print('test_a data shape: ',test_a.shape)
print('test_b data shape: ',test_b.shape)

#拼接数据
merge = pd.concat([train,test_b,test_a],ignore_index=True)

#增加行数列，方便后续特征文件的merge
merge['row_id'] = np.arange(len(merge))

del train
del test_a
gc.collect()

#商品类别提取
def extract_category(data):
    #data['cat_0'] = data['item_category_list'].str.split(';',expand=True)[0]
    data['cat_1'] = data['item_category_list'].str.split(';',expand=True)[1] 
    #data['cat_2'] = data['item_category_list'].str.split(';',expand=True)[2]
    data.drop('item_categroy_list',axis=1,inplace=True)
    gc.collect()
    return data
merge = extract_category(merge)

#id类特征处理
from sklearn.preprocessing import LabelEncoder

##删除context_id
merge.drop('context_id',axis=1,inplace=True)

##id类encoder
encoder_list = ['item_id','item_brand_id','item_city_id','cat_1','user_id','shop_id']

def encoder_category(data):
    for item in encoder_list:
        le = LabelEncoder()
        data[item] = le.fit_transform(data[item])
    del le
    gc.collect()
    return data

merge = encoder_category(merge)

#user年龄处理
merge['user_age_level'] = merge['user_age_level']%1000
merge.user_age_level = merge.user_age_level.replace(999,3)

#user职业处理
merge.user_occupation_id.replace(-1,2005,inplace=True)
merge.user_occupation_id %= 2000

#user星级处理
merge.user_star_level.replace(-1,3006,inplace=True)
merge.user_star_level %= 3000

#item_sales_level处理(填充缺失值-1)
#merge.item_sales_level.value_counts()
merge.item_sales_level.replace(-1,12,inplace=True)

#context_page_id处理
#merge.context_page_id.value_counts()
merge.context_page_id %= 4000

#shop_star_level处理
merge.shop_star_level %= 5000
merge.shop_star_level.replace(4999,13,inplace=True)

#提取时间（年、月份、日期、小时）
import time
def timestamp_datetime(value):
    formate = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(formate,value)
    return dt

def convert_time(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['day'] = data.day.apply(lambda x: 0 if x==31 else x)
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    data['minite'] = data.time.apply(lambda x: int(x[14:16]))
    #data['weekday'] = data['day']%7
    data.drop('context_timestamp',axis=1,inplace=True)
    return data

merge = convert_time(merge)


#生成基础特征文件
merge.to_csv('../feat_file/basic_feat.csv',index=False)
