import numpy as np
import pandas as pd
import gc

merge = pd.read_csv('../feat_file/basic_feat.csv')

#基础列数
basic_col_num = merge.shape[1]

#按时间排序，方便计算时差
merge['time'] = pd.to_datetime(merge['time'])
merge = merge.sort_values('time')

#用户当天对item_id、shop_id、item_brand_id、cat、context_page_id重复点击统计
def duplicate_same_day(data,item):
    subset = ['user_id', item, 'day']
    
    #提取重复点击item位置（首次点击、中间点击、最后点击）
    data['maybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 3

    #点击位置onehot
    features_trans = ['maybe']
    data = pd.get_dummies(data, columns=features_trans,prefix=item+'_maybe')
    data[item+'_maybe_0'] = data[item+'_maybe_0'].astype(np.int8)
    data[item+'_maybe_1'] = data[item+'_maybe_1'].astype(np.int8)
    data[item+'_maybe_2'] = data[item+'_maybe_2'].astype(np.int8)
    data[item+'_maybe_3'] = data[item+'_maybe_3'].astype(np.int8)

    #时间差Trick（与第一次点击和最后一次点击时间差）
    temp = data.loc[:,['time', 'user_id', item, 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'time': item+'_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data[item+'_diffTime_first'] = data['time'] - data[item+'_diffTime_first']

    temp = data.loc[:,['time', 'user_id', item, 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'time': item+'_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data[item+'_diffTime_last'] = data[item+'_diffTime_last'] - data['time']
        
    #填充缺失值，将timedelta转化为int
    data.loc[~data.duplicated(subset=subset, keep=False), [item+'_diffTime_first', item+'_diffTime_last']] = -1 
    data[item+'_diffTime_first'] = data[item+'_diffTime_first'].astype(int)
    data[item+'_diffTime_last'] = data[item+'_diffTime_last'].astype(int)
    
    #重复点击次数是否大于2
    temp=data.groupby(subset)['is_trade'].count().reset_index()
    temp.columns=['user_id', item, 'day',item+'_large2']
    temp[item+'_large2']=1*(temp[item+'_large2']>2)
    data = pd.merge(data, temp, how='left', on=subset)

    del temp
    gc.collect()
    return data

merge = duplicate_same_day(merge, 'item_id')
#merge = duplicate_item(merge, 'shop_id') #可能效果与item_id重复，无提升
merge = duplicate_same_day(merge, 'item_brand_id')
merge = duplicate_same_day(merge, 'cat_1')
merge = duplicate_same_day(merge, 'item_city_id')
merge = duplicate_same_day(merge, 'context_page_id')

#用户31～7所有天对item_id、shop_id、item_brand_id、cat、context_page_id重复点击统计
def duplicate_all_day(data,item):
    subset = ['user_id', item]
    
    #提取重复点击item位置（首次点击、中间点击、最后点击）
    data['maybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 3

    #点击位置onehot
    features_trans = ['maybe']
    data = pd.get_dummies(data, columns=features_trans,prefix=item+'_maybe_all_day')
    data[item+'_maybe_all_day_0'] = data[item+'_maybe_all_day_0'].astype(np.int8)
    data[item+'_maybe_all_day_1'] = data[item+'_maybe_all_day_1'].astype(np.int8)
    data[item+'_maybe_all_day_2'] = data[item+'_maybe_all_day_2'].astype(np.int8)
    data[item+'_maybe_all_day_3'] = data[item+'_maybe_all_day_3'].astype(np.int8)

    #时间差Trick（与第一次点击和最后一次点击时间差）
    temp = data.loc[:,['time', 'user_id', item]].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'time': item+'_diffTime_first_all_day'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data[item+'_diffTime_first_all_day'] = data['time'] - data[item+'_diffTime_first_all_day']

    temp = data.loc[:,['time', 'user_id', item]].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'time': item+'_diffTime_last_all_day'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data[item+'_diffTime_last_all_day'] = data[item+'_diffTime_last_all_day'] - data['time']
    
    #填充缺失值，将timedelta转化为int
    data.loc[~data.duplicated(subset=subset, keep=False), [item+'_diffTime_first_all_day', item+'_diffTime_last_all_day']] = -1 
    data[item+'_diffTime_first_all_day'] = data[item+'_diffTime_first_all_day'].astype(int)
    data[item+'_diffTime_last_all_day'] = data[item+'_diffTime_last_all_day'].astype(int)
    
    #重复点击次数是否大于2
    temp=data.groupby(subset)['is_trade'].count().reset_index()
    temp.columns=['user_id', item, item+'_large2_all_day']
    temp[item+'_large2_all_day']=1*(temp[item+'_large2_all_day']>2)
    data = pd.merge(data, temp, how='left', on=subset)

    del temp
    gc.collect()
    return data

#merge = duplicate_all_day(merge,'item_id')
merge = duplicate_all_day(merge,'shop_id')
merge = duplicate_all_day(merge,'cat_1')
#merge = duplicate_all_day(merge,'item_brand_id')
merge = duplicate_all_day(merge,'item_city_id')

#生成特征文件
new_cols = merge.columns.tolist()[basic_col_num:]
merge[['row_id']+new_cols].to_csv('../feat_file/duplicate_click.csv', index=False)
