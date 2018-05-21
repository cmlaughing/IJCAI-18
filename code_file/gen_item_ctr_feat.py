import numpy as np
import pandas as pd
import scipy.special as special
from collections import Counter
import gc

merge = pd.read_csv('../feat_file/basic_feat.csv')

#基础列数
basic_col_num = merge.shape[1]

#贝叶斯平滑

##贝叶斯平滑代码（基础精简版）
class BeyesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def update(self, imps, clks, iter_num,epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(
                imps,clks,self.alpha,self.beta)
            if(abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon):
                break
            print(new_alpha,new_beta,i)
            self.alpha = new_alpha
            self.beta = new_beta
            
    def __fixed_point_iteration(self, imps, clks,alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(beta))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))
            
        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)
    
##贝叶斯平滑代码（增加简单平滑方法）
class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C
    
    #与版本一相同
    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)
    
    #简单平滑，这个快一点
    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/tries[i])
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)

        return mean, var/(len(ctr_list)-1)
    
def BeyesCTR(data,item):
    item_all_list = list(set(data[item].values))
    print('开始统计'+item+'平滑')
    bs = HyperParam(1,1)
    dic_i = dict(Counter(data[data.is_trade.notnull()][item].values))
    dic_cov = dict(Counter(data[data.is_trade==1.0][item].values))
    l = list(dic_i.keys())
    I=[]
    C=[]
    for item_id in l:
        I.append(dic_i[item_id])
    for item_id in l:
        if item_id not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[item_id])
    print('开始平滑操作')
    bs.update_from_data_by_moment(I, C)
    print(bs.alpha,bs.beta)
    dic_PH = {}
    for pos in item_all_list:
        if pos not in dic_i:
            dic_PH[pos] = (bs.alpha)/(bs.alpha+bs.beta)
        elif pos not in dic_cov:
            dic_PH[pos] = (bs.alpha)/(dic_i[pos]+bs.alpha+bs.beta)
        else:
            dic_PH[pos] = (dic_cov[pos]+bs.alpha)/(dic_i[pos]+bs.alpha+bs.beta)
    df_out = pd.DataFrame({item:list(dic_PH.keys()),
                           'PH_'+item:list(dic_PH.values())})
    return df_out    

##使用6号之前的数据统计item 转化率（6号ctr与前几天相比太低，不加入计算）
def getBeyesCtr(data,item):
    statistic_data = data[data.day<6]
    ph_cat = BeyesCTR(statistic_data,item)
    data = data.merge(ph_cat,how='left',on=item)
    return data

PH_item_list = ['item_id','item_brand_id','item_city_id','shop_id'] #user_id点击次数太少，未计算平滑转化率
for item in PH_item_list:
    merge = getBeyesCtr(merge,item)
    print('PH {} finished'.format(item))
    
    
#生成特征文件
new_cols = merge.columns.tolist()[basic_col_num:]
merge[['row_id']+new_cols].to_csv('../feat_file/duplicate_click.csv', index=False)    


