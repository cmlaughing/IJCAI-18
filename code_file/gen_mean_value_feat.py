import numpy as np
import pandas as pd

merge = pd.read_csv('../feat_file/basic_feat.csv')

#基础列数
basic_col_num = merge.shape[1]

#用户年龄、星级均值

##年龄均值
groupby = merge.groupby('item_id')['user_age_level'].mean().reset_index(name='item_mean_age')
merge = merge.merge(groupby,how='left',on='item_id')
groupby = merge.groupby('shop_id')['user_age_level'].mean().reset_index(name='shop_mean_age')
merge = merge.merge(groupby,how='left',on='shop_id')
groupby = merge.groupby('item_brand_id')['user_age_level'].mean().reset_index(name='brand_mean_age')
merge = merge.merge(groupby,how='left',on='item_brand_id')
groupby = merge.groupby('cat_1')['user_age_level'].mean().reset_index(name='cat_mean_age')
merge = merge.merge(groupby,how='left',on='cat_1')
groupby = merge.groupby('hour')['user_age_level'].mean().reset_index(name='hour_mean_age')
merge = merge.merge(groupby,how='left',on='hour')


##星级均值
groupby = merge.groupby('shop_id')['user_star_level'].mean().reset_index(name='shop_mean_user_star')
merge = merge.merge(groupby,how='left',on='shop_id')
groupby = merge.groupby('item_id')['user_star_level'].mean().reset_index(name='item_mean_user_star')
merge = merge.merge(groupby,how='left',on='item_id')
groupby = merge.groupby('item_brand_id')['user_star_level'].mean().reset_index(name='brand_mean_user_star')
merge = merge.merge(groupby,how='left',on='item_brand_id')


#item_id/shop_id 属性均值（这里属性指 item_price_level、shop_review_num_level等，不是property）

##item_id/shop_id 属性均值 (考虑到这些item属性值会变化，做一个均值统计）
for item in ['item_price_level','item_sales_level','item_pv_level','item_collected_level']:
    groupby = merge.groupby('item_id')[item].mean().reset_index(name=item+'_mean_value')
    merge = merge.merge(groupby,how='left',on='item_id')
    print(item + ' finished')
    
for item in ['shop_review_num_level','shop_review_positive_rate','shop_star_level',
             'shop_score_service','shop_score_delivery','shop_score_description']:
    groupby = merge.groupby('shop_id')[item].mean().reset_index(name=item+'_mean_value')
    merge = merge.merge(groupby,how='left',on='shop_id')
    print(item+' finished')

##用户点击的所有item_id/shop_id 属性均值 
item_list = ['item_price_level','item_sales_level','item_collected_level','item_pv_level',
             'shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service',
            'shop_score_delivery','shop_score_description']

for item in item_list:
    groupby = merge.groupby('user_id')[item].mean().reset_index(name='user_mean_'+item)
    merge = merge.merge(groupby,how='left',on='user_id')
    print(item+' finished')
    
#生成特征文件
new_cols = merge.columns.tolist()[basic_col_num:]
merge[['row_id']+new_cols].to_csv('../feat_file/mean_value_feat.csv',index=False)
