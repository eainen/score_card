# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:50:23 2017

@author: Daniel
"""

"""
Score all the observations per the final model, and calculate:
1. K-S 
2. Gini
3. Lift

"""

#%% import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#%% metrics functions
def scorebucket(X, y, lr_model, lr_res, bins=20):
    
    model_params = lr_res.params    
    ln_odds = (X * model_params).sum(axis=1)
    #score_train = A - B * ln_odds_train   
    prob = 1/(np.exp(-ln_odds)+1)
    prob.name = 'Prob'
    prob = pd.DataFrame(prob)
    
    prob['Target'] = y 
    prob.sort_values(by='Prob', ascending=False, inplace=True)
    prob['Rank'] = 1
    prob.Rank = prob.Rank.cumsum()
    prob['Bucket'] = pd.qcut(prob.Rank, bins)
    
    return prob

def ksdistance(prob):   
    bucket = prob.groupby('Bucket',
                          as_index=False)['Target'].agg({'Total':'count',
                                                         'Bad':'sum',
                                                         'BadRate':'mean'})
    bucket.drop('Bucket', axis=1, inplace=True)
    bucket.eval('Good = Total - Bad', inplace=True)
    bucket['CumBad'] = bucket.Bad.cumsum()
    bucket['CumGood'] = bucket.Good.cumsum()
    bucket['CumTotal'] =bucket.Total.cumsum()
    bucket['PctCumBad'] = bucket.CumBad/bucket.CumBad.max()
    bucket['PctCumGood'] = bucket.CumGood/bucket.CumGood.max()
    bucket['PctCumTotal'] = bucket.CumTotal/bucket.CumTotal.max()
    bucket['KS'] = bucket.PctCumBad - bucket.PctCumGood
    bucket.eval('Lift = PctCumBad/PctCumTotal', inplace=True)
    
    # KS Distance
    metric_ks = bucket.KS.max()
    
    bucket[['PctCumBad',
            'PctCumGood',
            'KS']].plot(style=['r','b','g'], ylim=[0,1],
                        title='KS Distance = %0.4f'%metric_ks)
    #plt.ylim([0,1])
    plt.xlim([0,bucket.shape[0]])
    plt.xlabel('Score Buckets')
    plt.ylabel('Pct Distribution')
    plt.show() 
    return metric_ks, bucket

    
def aucroc(X, y, lr_model, lr_res):
    # AUC-ROC and Gini
    y_hat = lr_model.predict(lr_res.params, X[lr_res.params.index])
    
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_hat)
    metric_auc = auc(false_positive_rate, true_positive_rate)
    metric_gini = metric_auc * 2 - 1 
    
    plt.title("Receiver Operating Characteristic")
    plt.plot(false_positive_rate, true_positive_rate, 'b', 
             label='AUC = %0.4f\nGini = %0.4f' % (metric_auc,metric_gini))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
        
    return metric_auc, metric_gini


def PSI(prob_train, prob_valid, bins=10):
    """
    Calculate model PSI.
    
    INPUTs:
    > prob_train, prob_valid: model predicted probabilities (or scores) of trainning set and validating set
    > bins: number of score buckets, default is 10
    
    OUTPUTs:
    > p_stab: output dataframe
    > si: the stability index number (sum of all) 
    """
    # get the score bins
    _, p_bins = pd.cut(prob_train.Prob, bins=bins, retbins=True)
    p_bins[0] = 0
    p_bins[-1] = 1
    # cut the training set
    prob_train['p_range'] = pd.cut(prob_train.Prob, p_bins, include_lowest=True)
    p_range_train = prob_train.groupby('p_range').Target.count()
    # cut the validation set
    prob_valid['p_range'] = pd.cut(prob_valid.Prob, p_bins, include_lowest=True)
    p_range_valid = prob_valid.groupby('p_range').Target.count()
    # generate stability index table
    p_stab = pd.DataFrame(p_range_train)
    p_stab.columns = ['Train'] 
    p_stab['Validate'] = p_range_valid
    # calculate index
    p_stab['PctTrain'] = p_stab['Train'] / p_stab['Train'].sum() 
    p_stab['PctValidate'] = p_stab['Validate'] / p_stab['Validate'].sum() 
    p_stab.eval('PctDiff = PctValidate - PctTrain', inplace=True)
    p_stab['LnDiff'] = np.log(p_stab['PctValidate']/p_stab['PctTrain'])
    p_stab.eval('StabIdx = LnDiff * PctDiff', inplace=True)
    si = p_stab.StabIdx.sum()
    
    return p_stab, si

def PSI_qcut(prob_train, prob_valid, bins=10):
    """
    Calculate model PSI.
    
    INPUTs:
    > prob_train, prob_valid: model predicted probabilities (or scores) of trainning set and validating set
    > bins: number of score buckets, default is 10
    
    OUTPUTs:
    > p_stab: output dataframe
    > si: the stability index number (sum of all) 
    """
    # get the score bins
    _, p_bins = pd.qcut(prob_train.Prob, q=bins, retbins=True)
    p_bins[0] = 0
    p_bins[-1] = 1
    # cut the training set
    prob_train['p_range'] = pd.cut(prob_train.Prob, p_bins, include_lowest=True)
    p_range_train = prob_train.groupby('p_range').Target.count()
    # cut the validation set
    prob_valid['p_range'] = pd.cut(prob_valid.Prob, p_bins, include_lowest=True)
    p_range_valid = prob_valid.groupby('p_range').Target.count()
    # generate stability index table
    p_stab = pd.DataFrame(p_range_train)
    p_stab.columns = ['Train'] 
    p_stab['Validate'] = p_range_valid
    # calculate index
    p_stab['PctTrain'] = p_stab['Train'] / p_stab['Train'].sum() 
    p_stab['PctValidate'] = p_stab['Validate'] / p_stab['Validate'].sum() 
    p_stab.eval('PctDiff = PctValidate - PctTrain', inplace=True)
    p_stab['LnDiff'] = np.log(p_stab['PctValidate']/p_stab['PctTrain'])
    p_stab.eval('StabIdx = LnDiff * PctDiff', inplace=True)
    si = p_stab.StabIdx.sum()
    
    return p_stab, si 


##########################################################
#  以下为变量SSI计算函数
# 使用woeTransform的函数，将原始值转为分组clus_num
def get_clus_n(raw, bin_tbl):
    lbound = bin_tbl.LBound.values
    ubound = bin_tbl.UBound.values
    tbl_idx = 0
    for i in range(len(ubound)):
        # NA handler 
        if np.isnan(raw) and np.isnan(lbound[i]):
            tbl_idx = i 
        else: 
            if ~np.isnan(lbound[i]) and raw<=ubound[i] and raw>lbound[i]:
                tbl_idx = i
            else: continue
    return bin_tbl.ix[tbl_idx, 'clus_num']

def get_clus(data, bin_info_n, bin_info_c, var_dict):
    
    clus_data = pd.DataFrame()
    i = 1
    
    for var in var_dict.VAR_NAME:
    
        if var_dict[var_dict.VAR_NAME==var]['TYPE'].values == 'NUM':
            bin_tbl = bin_info_n[bin_info_n.var_name == var]
            bin_tbl.reset_index(drop=True, inplace=True)
            clus_data['clus_'+var] = data[var].apply(lambda x: get_clus_n(x,bin_tbl))
            
            #print('Transformed: %s \t %d/%d' % (var, i, var_dict.shape[0]))
            
        elif var_dict[var_dict.VAR_NAME==var]['TYPE'].values == 'CAT':

            bin_tbl = bin_info_c[bin_info_c.var_name==var]
            bin_tbl = bin_tbl[['category','clus_num']]
            bin_tbl.set_index('category', inplace=True)
            # pd.Series.map for series only
            map_series = pd.Series(bin_tbl.clus_num.values, index=bin_tbl.index.astype(str))
            clus_data['clus_'+var] = data[var].apply(str).map(map_series)
            
            #print('Transformed: %s \t %d/%d' % (var, i, var_dict.shape[0]))
            
        else: break
    
        i += 1
    
    return clus_data 


def SSI(lr_res, data_valid, bin_info_n, bin_info_c, var_dict):
    """
    Map raw data into clus_num as per bin_info tables, identify type (num or cat) as per var_dict;
    Compare PctTotal of the validation set with the training set;
    Calculate stability index for each vairable in the model.
    
    INPUTS:
    > lr_res - the fitted logistic regression model result (statsmodels object)
    > data_valid - dataframe of raw validation data
    > bin_info_n - binning results of numercical variables
    > bin_info_c - binning results of categorical variables
    > var_dict - variable types table from excel file
    
    OUTPUT:
    > dataframe contraining stability indices of all model variables
    """
    # this could take a while depending on the data size 
    # and the number of variables
    clus_valid = get_clus(data_valid, bin_info_n, bin_info_c, var_dict)
    # exclude Intercept
    params = lr_res.params.index.tolist()[1:]
    # create an empy dataframe to store reults
    clus_all = pd.DataFrame(columns=['var_name', 'clus_num', 'train', 'validate'])
    
    for name in params:
        # parameter name starts with 'clus_', remove it to get the original name
        param = '_'.join(name.split('_')[1:])
        
        if param in bin_info_n.var_name.unique():
            
            bin_info = bin_info_n
            # create new dataframe with clus_num as index
            clus = pd.DataFrame(clus_valid.groupby('clus_'+param, as_index=True).count().iloc[:,0])
            clus.columns = ['validate']
            clus.index.rename('clus_num', inplace=True)
            clus['var_name'] = param
            # population distributions as in the binning result from trainning data
            clus['train'] = bin_info[bin_info.var_name==param].groupby('clus_num')['PctTotal'].sum()/100
            clus['validate'] = clus['validate'] / clus['validate'].sum()
            clus.reset_index(inplace=True)
            clus = clus[['var_name', 'clus_num', 'train', 'validate']]
            # calculate stability indices
            clus.eval('pct_diff = validate - train', inplace=True)
            clus['ln_diff'] = np.log(clus['validate']/clus['train'])
            clus.eval('stab_idx = pct_diff * ln_diff', inplace=True)
            clus['idx_sum'] = clus['stab_idx'].sum()
            
            clus_all = clus_all.append(clus)
            clus_all = clus_all[['var_name', 'clus_num', 'train', 'validate', 'pct_diff', 'ln_diff', 'stab_idx', 'idx_sum']]
            clus_all.reset_index(inplace=True)

        elif param in bin_info_c.var_name.unique():
            # same as above, but for categorical bin_info table - different layout
            bin_info = bin_info_c
            clus = pd.DataFrame(clus_valid.groupby('clus_'+param, as_index=True).count().iloc[:,0])
            clus.columns = ['validate']
            clus.index.rename('clus_num', inplace=True)
            clus['var_name'] = param
            # the only difference here is to use max() instead of sum()
            clus['train'] = bin_info[bin_info.var_name==param].groupby('clus_num')['PctTotal'].max()/100
            clus['validate'] = clus['validate'] / clus['validate'].sum()
            clus.reset_index(inplace=True)
            clus = clus[['var_name', 'clus_num', 'train', 'validate']]
            clus.eval('pct_diff = validate - train', inplace=True)
            clus['ln_diff'] = np.log(clus['validate']/clus['train'])
            clus.eval('stab_idx = pct_diff * ln_diff', inplace=True)
            clus['idx_sum'] = clus['stab_idx'].sum()

            clus_all = clus_all.append(clus)
            clus_all = clus_all[['var_name', 'clus_num', 'train', 'validate', 'pct_diff', 'ln_diff', 'stab_idx', 'idx_sum']]
            clus_all.reset_index(inplace=True)
            
        else: 
            continue
            
    return clus_all

def ks_bin(y_pred, y_true, bins=10, ks_plot=False):
    prob=pd.DataFrame(y_pred)
    prob['Target'] =y_true 
    prob.columns=["Prob","Target"]
    prob.sort_values(by='Prob', ascending=False, inplace=True)
    prob['Rank'] = 1
    prob.Rank = prob.Rank.cumsum()
    prob['Bucket'] = pd.qcut(prob.Rank, bins)  
    
    bucket = prob.groupby('Bucket',
                          as_index=False)['Target'].agg({'Total':'count',
                                                         'Bad':'sum',
                                                         'BadRate':'mean'})
    bucket.drop('Bucket', axis=1, inplace=True)
    bucket.eval('Good = Total - Bad', inplace=True)
    bucket['CumBad'] = bucket.Bad.cumsum()
    bucket['CumGood'] = bucket.Good.cumsum()
    bucket['CumTotal'] =bucket.Total.cumsum()
    bucket['PctCumBad'] = bucket.CumBad/bucket.CumBad.max()
    bucket['PctCumGood'] = bucket.CumGood/bucket.CumGood.max()
    bucket['PctCumTotal'] = bucket.CumTotal/bucket.CumTotal.max()
    bucket['KS'] = bucket.PctCumBad - bucket.PctCumGood
    bucket.eval('Lift = PctCumBad/PctCumTotal', inplace=True)
    
    # KS Distance
    metric_ks = bucket.KS.max()
    if ks_plot:
        bucket[['PctCumBad',
            'PctCumGood',
            'KS']].plot(style=['r','b','g'], ylim=[0,1],
                        title='KS Distance = %0.4f'%metric_ks)

        plt.xlim([0,bucket.shape[0]])
        plt.xlabel('Score Buckets')
        plt.ylabel('Pct Distribution')
        plt.show()
    return metric_ks, bucket