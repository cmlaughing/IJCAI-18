import numpy as np
import pandas as pd

merge = pd.read_csv('../feat_file/basic_feat.csv')

#基础特征列数
col_num = merge.shape[1]

#按时间排序，方便统计cumcount
merge = merge.sort_values('time')

#用户-商品当天、小时、分钟点击次数（截止当前时刻）
def user_item_cumcount(data,item):
    user_item_day_cumcount = data.groupby(['user_id','day',item]).cumcount(
                                ).to_frame().rename(column={0:'user_'+item+'_day_cumcount})
    data = pd.concat([data,user_item_day_cumcount],axis=1)
    user_item_hour_cumcount = data.groupby(['user_id','day','hour',item]).cumcount(
                                ).to_frame().rename(columns={0:'user_'+item+'_hour_cumcount'})
    data = pd.concat([data,user_item_hour_cumcount],axis=1)
    user_item_minite_cumcount = data.groupby(['user_id','day','hour','minite',item]).cumcount(
                                ).to_frame().rename(columns={0:'user_'+item+'_minite_cumcount'})    
    data = pd.concat([data,user_item_minite_cumcount],axis=1)
    return data
                                                        
merge = user_item_cumcount(merge,'item_id')
#merge = user_item_cumcount(merge,'shop_id')    
 
#用户-商品/商店当天、小时、分钟点击次数 （全天统计）
def user_item_count(data,item):
    user_item_day_count = data.groupby(['user_id','day',item]
                                      ).size().reset_index(name='user_day_active_'+item)
    data = data.merge(user_item_day_count,on=['user_id','day',item],how='left')
    user_item_hour_count = merge.groupby(['user_id','day','hour',item]
                                     ).size().reset_index(name='user_hour_active_'+item)
    data = data.merge(user_item_hour_count,on=['user_id','day','hour',item],how='left')
    user_item_minite_count = merge.groupby(['user_id','day','hour','minite',item]
                                      ).size().reset_index(name='user_minite_active_'+item)
    data = data.merge(user_item_minite_count,on=['user_id','day','hour','minite',item],how='left')
    return data    
                                                            
merge = user_item_count(merge, 'item_id')
merge = user_item_count(merge, 'shop_id')
#merge = user_item_count(merge, 'item_brand_id')                                                            
#merge = user_item_count(merge, 'cat_1')                                                             
                                                            
#用户当天、小时、分钟点击不同商品/商店个数
def user_item_nunique(data,item):
    user_item_day_nunique = data.groupby(['user_id','day']
                                      )[item].nunique().reset_index(name='user_day_nunique_'+item)
    data = data.merge(user_item_day_nunique,on=['user_id','day'],how='left')
    user_item_hour_nunique = merge.groupby(['user_id','day','hour']
                                     )[item].nunique().reset_index(name='user_hour_nunique_'+item)
    data = data.merge(user_item_day_nunique,on=['user_id','day','hour'],how='left')
    user_item_minite_nunique = merge.groupby(['user_id','day','hour','minite']
                                      )[item].nunique().reset_index(name='user_minite_nunique_'+item)
    data = data.merge(user_item_day_nunique,on=['user_id','day','hour','minite'],how='left')
    return data 

merge = user_item_nunique(merge, 'item_id')
merge = user_item_nunique(merge, 'shop_id')
#merge = user_item_count(merge, 'item_brand_id')                                                            
#merge = user_item_count(merge, 'cat_1')                                                            

#item 当天、小时、分钟被点击次数（截止当前时刻）                                                             
def item_cumcount(data,item_list): 
    for item in item_list:
        print(item+' start')
        user_item_day_cumcount = data.groupby([item,
            'day']).cumcount().to_frame().rename(columns={0:item+'_day_cumcount'})
        data = pd.concat([data,user_item_day_cumcount],axis=1)
        user_item_hour_cumcount = data.groupby([item,'day',
            'hour']).cumcount().to_frame().rename(columns={0:item+'_hour_cumcount'})
        data = pd.concat([data,user_item_hour_cumcount],axis=1)
        user_item_minite_cumcount = data.groupby([item,'day',
            'hour','minite']).cumcount().to_frame().rename(columns={0:item+'_minite_cumcount'})    
        data = pd.concat([data,user_item_minite_cumcount],axis=1)
        print(item+' has finished') 
    return data                                                    
                                                            
item_list = ['item_id','item_brand_id','shop_id']
merge = item_cumcount(merge,item_list)
                                                            
#item日均活跃用户数
def day_mean_user(data,item):
    item_day_active_user =data.groupby([item,'day'])['user_id'].nunique().reset_index(name=item+'_day_user_count')
    item_day_mean_user = item_day_active_user.groupby(item)[item+'_day_user_count'].mean().reset_index(name=item+'_day_mean_user')
    data = data.merge(item_day_mean_user,on=[item],how='left')
    return data
    
for item in ['item_id','shop_id','item_brand_id','item_city_id','cat_1','hour']:
    merge = day_mean_user(merge,item)
                                                            
#生成特征文件                                                            
new_cols = merge.columns.tolist()[basic_col_num:]                                                            
merge[['row_id']+new_cols].to_csv('../click_count_feat.csv',index=False)                                                            
                                                            
