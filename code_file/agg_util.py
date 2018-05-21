#一些常用的聚合统计函数包装
import pandas as pd


#根据columns分组，统计每组数量
def feat_count(df,columns,cname):
    add = df.groupby(columns)size().reset_index(name=cname)
    df=df.merge(add,on=columns,how="left")
    return df

#根据columns分组，统计组内不同value个数
def feat_nunique(df,columns,value,cname):
    add = df.groupby(columns)[value].nunique().reset_index(name=cname)
    df=df.merge(add,on=columns,how="left")
    return df

#根据columns分组， 统计组内value的中位数
def feat_median(df,columns,value,cname):
    add = df.groupby(columns)[value].median().reset_index(name=cname)
    df=df.merge(add,on=columns,how="left")
    return df

#根据columns分组， 统计组内value的平均值
def feat_mean(df,columns,value,cname):
    add = df.groupby(columns)[value].mean().reset_index(name=cname)
    df=df.merge(add,on=columns,how="left")
    return df

#根据columns分组， 统计组内value的和
def feat_sum(df,columns,value,cname):
    add = df.groupby(columns)[value].sum().reset_index(name=cname)
    df=df.merge(add,on=columns,how="left")
    return df

#根据columns分组，统计组内value的最大值
def feat_max(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index(name=cname)
   	df=df.merge(add,on=columns,how="left")
    return df

#根据columns分组， 统计组内value的最小值
def feat_min(df,columns,value,cname):
    add = df.groupby(columns)[value].min().reset_index(name=cname)
    df=df.merge(add,on=columns,how="left")
   	return df

#根据columns分钟， 统计组内value的标准差
def feat_std(df,columns,value,cname):
    add = df.groupby(columns)[value].std().reset_index(name=cname)
    df=df.merge(add,on=columns,how="left")
	return df
