# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:00:44 2017

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

###############################################################################
def nullClass(var, feature, target):
    """
    Identify null values in the data, set aside a separate class
    to be appended at the end of the classed table for numerics
    """
    #var = data[[feature, target]]
    var_nan = var[var[feature].isnull()]
    var_n = var[-var[feature].isnull()]
    grouped_nan = pd.DataFrame({'Class': 'NA',
                                'LBound': None,
                                'UBound': None,
                                'Total': var_nan[feature].shape[0],
                                'PctTotal': None,
                                'Bad': var_nan[target].sum(),
                                'PctBad': None,
                                'BadRate': None,
                                'Good': None,
                                'PctGood': None,
                                'GoodRate': None,
                                'WOE': None,
                                'IV': None,},
                                 index=['NA'])
    grouped_nan.eval('BadRate = Bad / Total', inplace=True)
    grouped_nan.eval('GoodRate = 1 - BadRate', inplace=True)
    grouped_nan.eval('Good = Total - Bad', inplace=True)
    #grouped_nan.set_index('Class', inplace=True)

    return var_n, grouped_nan

###############################################################################

def rawCut(data, feature, target):
    """
    raw cut the input numeric data with boundaries as the unique values
    and calculate scores for pooling steps
    """
    data1 = data[[feature, target]]
    # sort the data in ascending order
    data1.sort_values(feature, inplace=True)
    # set boundaries to be all unique values
    intervals = data1[feature].unique().tolist()
    intervals = sorted(intervals)
    # add a inf lower boundary to prevent the first 2 adjacent uniques merging
    intervals.insert(0, -np.inf)
    # split the data by the inital boundaries
    data1['Class'] = pd.cut(data1[feature], intervals, include_lowest=True)
    grouped = data1.groupby('Class',
                           as_index=False)[target].agg({'Total':'count',
                                                       'Bad':'sum',
                                                       'BadRate':'mean'})
    bounds = data1.groupby('Class',
                         as_index=False)[feature].agg({'LBound':'min',
                                                        'UBound':'max'})

    # check 0 bad/good counts and pop the boundaries
    to_pop_idx = grouped[(grouped.BadRate==0)|(grouped.BadRate==1)].index.tolist()
    if len(to_pop_idx) == 0:
        pass
    else:
        to_pop_val = []
        for idx in to_pop_idx:
            to_pop_val.append(intervals[idx])
        for val in to_pop_val:
            intervals.remove(val)
        if intervals[0]==-np.inf:
            pass
        else:
            intervals.insert(0, -np.inf)
        # re-group
        data1['Class'] = pd.cut(data1[feature], intervals, include_lowest=False)
        grouped = data1.groupby('Class',
                               as_index=False)[target].agg({'Total':'count',
                                                           'Bad':'sum',
                                                           'BadRate':'mean'})
        bounds = data1.groupby('Class',
                             as_index=False)[feature].agg({'LBound':'min',
                                                            'UBound':'max'})

    grouped['LBound'] = bounds['LBound']
    grouped['UBound'] = bounds['UBound']
    grouped.eval('Good = Total - Bad', inplace=True)
    grouped.eval('GoodRate = 1 - BadRate', inplace=True)
    grouped['PctTotal'] = grouped['Total'] / grouped['Total'].sum() * 100
    grouped['PctGood'] = grouped['Good'] / grouped['Good'].sum() * 100
    grouped['PctBad'] = grouped['Bad'] / grouped['Bad'].sum() * 100
    grouped['WOE'] = np.log(grouped['PctBad']/grouped['PctGood'])
    grouped.eval('IV = (PctBad - PctGood) * WOE / 100', inplace=True)

    ordered =  grouped[['Class','LBound','UBound','Total','PctTotal',
                        'Bad','PctBad','BadRate','Good','PctGood',
                        'GoodRate','WOE','IV']]
    output = poolingScores(ordered)

    return output, intervals


###############################################################################
def ClassContinuous(data,feature,target,method='qcut',bins=10,cutoffs=None):
    """
    Fine classing for
    continuous variables

    INPUT:
    data = dataframe with features and target
    feature = name of feature column as string
    target = name of target column as string

    OUTPUT:
    classed dataframe
    """
    # turn off pandas warning messeges 
    pd.options.mode.chained_assignment = None  # default='warn'

    data1 = data[[feature, target]]
    if method == 'qcut': # cut by number of bins quantile
        data1['Class'] = pd.qcut(data1[feature], bins)
        cutoffs_ret = pd.qcut(data1[feature], bins, retbins=True)[1]
    elif method == 'cut': # cut by cutoff values
        data1['Class'] = pd.cut(data1[feature], cutoffs, include_lowest=True)
        cutoffs_ret = cutoffs
    else:
        '''
        to-do:
            write an error handler
        '''
        return "ERROR: Unknown cut method"

    #data1['Class_id'] = pd.factorize(data1.Class)[0] + 1

    grouped = data1.groupby('Class',
                            as_index=False)[target].agg({'Total':'count',
                                                         'Bad':'sum',
                                                         'BadRate':'mean'})
    bounds = data1.groupby('Class',
                            as_index=False)[feature].agg({'LBound':'min',
                                                         'UBound':'max'})

    #grouped.index = grouped['Class']
    grouped['LBound'] = bounds['LBound']
    grouped['UBound'] = bounds['UBound']
    grouped.eval('Good = Total - Bad', inplace=True)
    grouped.eval('GoodRate = 1 - BadRate', inplace=True)
    grouped['PctTotal'] = grouped['Total'] / grouped['Total'].sum() * 100
    grouped['PctGood'] = grouped['Good'] / grouped['Good'].sum() * 100
    grouped['PctBad'] = grouped['Bad'] / grouped['Bad'].sum() * 100
    grouped['WOE'] = np.log(grouped['PctBad']/grouped['PctGood'])
    grouped.eval('IV = (PctBad - PctGood) * WOE / 100', inplace=True)

    ordered =  grouped[['Class','LBound','UBound','Total','PctTotal',
                        'Bad','PctBad','BadRate','Good','PctGood',
                        'GoodRate','WOE','IV']]
    output = poolingScores(ordered)

    return output, cutoffs_ret

###############################################################################
def ClassDiscrete(data, feature, target, ordered=False, mapping=None):
    
    pd.options.mode.chained_assignment = None  # default='warn'

    data = data[[feature, target]]
    data[feature] = data[feature].apply(str)

    if mapping != None:
        data['clus'] = data[feature].apply(lambda x: mapping[x])
        grouped = data.groupby(by='clus',
                               as_index=False)[target].agg({'Total':'count',
                                                            'Bad':'sum',
                                                            'BadRate':'mean'})
      
    else:
        grouped = data.groupby(by=feature,
                               as_index=False)[target].agg({'Total':'count',
                                                           'Bad':'sum',
                                                           'BadRate':'mean'})

    grouped.eval('Good = Total - Bad', inplace=True)
    grouped.eval('GoodRate = 1 - BadRate', inplace=True)
    grouped['PctTotal'] = grouped['Total'] / grouped['Total'].sum() * 100
    grouped['PctGood'] = grouped['Good'] / grouped['Good'].sum() * 100
    grouped['PctBad'] = grouped['Bad'] / grouped['Bad'].sum() * 100
    grouped['WOE'] = np.log(grouped['PctBad']/grouped['PctGood'])
    grouped.eval('IV = (PctBad - PctGood) * WOE / 100', inplace=True)

    if ordered:
        sort = grouped.sort_values('BadRate')
        output = poolingScores(sort)
        return output
    else:
        output = poolingScores(grouped)
    return output

###############################################################################
def poolingScores(var):
    """
    Calculate columns for pooling algorithms
    """
    # F scores for adjacent pooling
    var['F_standalone'] = (var['IV'] + var['IV'].shift(1)) * 10000
    var['F_combine'] = np.log((var['PctBad'] + var['PctBad'].shift(1)) \
                            / (var['PctGood'] + var['PctGood'].shift(1))) \
                            * (var['PctBad'] + var['PctBad'].shift(1) \
                            - var['PctGood'] - var['PctGood'].shift(1))*100
    var.eval('F_loss = - (F_combine - F_standalone)', inplace=True)
    # cumulative bad rates for monotone adjacent pooling
    var['CumBad'] = var['Bad'].cumsum()
    var['CumTotal'] = var['Total'].cumsum()
    var.eval('CumBadRate = CumBad / CumTotal', inplace=True)
    return var

###############################################################################
def plotBinning(var, value='BadRate'):
    """
    plots for coarse classing
    """
    font = fm.FontProperties(fname='c:\\windows\\fonts\\simhei.ttf',size=12)

    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    ax1a = var[value].plot.line(style='r')
    ax1b = var['PctTotal'].plot.bar(secondary_y=True, alpha=0.3, 
                                    grid=True, title=value)
    ax1a.set_xticklabels(var.index, fontproperties=font)

    plt.subplot(122)
    ax2a = var.F_loss.plot.line(style='gx-')
    ax2b = var.IV.plot.bar(secondary_y=True, color='g', alpha = 0.5,
                    grid=True, title=('IV: %f' % var[var.IV != np.inf].IV.sum()))
    ax2a.set_xticklabels(var.index, fontproperties=font)

    plt.tight_layout()
    plt.show()

###############################################################################
def plotBinning_n(bin_info_n,var_name):
    """
    plots for coarse classing
    """
     
    tbl=bin_info_n[bin_info_n.var_name==var_name]
    IV=tbl.IV.sum()
    print "IV= ",IV
    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plt.plot(tbl.clus_num,tbl["BadRate"],label='BadRate',linewidth=1.5,color='r',marker='o', markerfacecolor='blue',markersize=6) 
    plt.xticks(tbl.clus_num, tbl.bucket, rotation=90)
    #ax1 = tbl["BadRate"].plot.line(style='r',grid=True,title=var_name)
    #ax1.set_xticklabels(tbl.bucket)
    plt.subplot(122)
    ax1 = tbl['PctTotal'].plot.bar( alpha=0.3,grid=True, title=var_name)
    ax1.set_xticklabels(tbl.bucket)
    plt.show()
    
###############################################################################
def NumAutoBin(raw_data, var_name, target, tol=100, min_pct=5, 
               min_bins=2, max_bins=10, init_bins=50):
    
    
    pd.options.mode.chained_assignment = None  # default='warn'
    
    var = raw_data[[var_name, target]]

    count = 0
    
    # handle missing values
    null_count = var[var[var_name].isnull()].shape[0]
    if null_count > 0:
        # store null class in grouped_null
        var, grouped_null = nullClass(var, var_name, target)
    else:
        pass
    
    var_g, cutoffs = rawCut(var, var_name, target)
    ttl_bound = len(cutoffs)
    
    if ttl_bound > 500:
        bins = init_bins
        while bins > 0:
            try:
                var_g, cutoffs = ClassContinuous(var, var_name, target, 
                                                 'qcut', bins)
                cutoffs = cutoffs.tolist()
                break
            except ValueError:
                bins -= 1
            
    else:
        pass
    
    # adjacent pooling steps (not monotone!)
    while var_g.shape[0] > min_bins:
        # reset index to get proper position
        var_g.reset_index(inplace=True)
        
        # first work on the inf IVs
        if var_g.IV.max() == np.inf: 
            #idx_inf = var_g.IV.idxmax()  # this bypasses the inf !!!
            idx_inf = var_g[var_g.IV == np.inf].index[0] # this one is okay
            cutoffs.pop(int(idx_inf));
            #print("removed empty interval")
        # then locate the minimum F_loss and merge bins
        else:

            idx_min = var_g.F_loss.idxmin()
            cutoffs.pop(int(idx_min));
            #print("{} / {}".format(len(cutoffs), ttl_bound))
       
        # recalculate with new cutoffs
        var_g, _ = ClassContinuous(data=var, feature=var_name, 
                                   target=target, method='cut', 
                                   cutoffs=cutoffs)
        count += 1
        
        # break conditions
        if var_g.PctTotal.min()>=min_pct:
            break
        elif var_g.F_loss.min()>=tol and var_g.shape[0]<=max_bins:
            break
        elif var_g.shape[0]<=min_bins:
            break
        else: 
            continue
            
    if cutoffs[0]== -np.inf:
        pass;
    else:
        cutoffs.insert(0,-np.inf)
        var_g, _ = ClassContinuous(data=var, feature=var_name, 
                                   target=target, method='cut', 
                                   cutoffs=cutoffs)
        count += 1
        
    if null_count > 0:
        #append the null class, recalculate metrics
        var_g = var_g.append(grouped_null)
        #recalculate PctTotal, PctBad, PctGood, WOE, IV
        var_g['PctTotal'] = var_g['Total'] / var_g['Total'].sum() * 100
        var_g['PctGood'] = var_g['Good'] / var_g['Good'].sum() * 100
        var_g['PctBad'] = var_g['Bad'] / var_g['Bad'].sum() * 100
        var_g['WOE'] = np.log(var_g['PctBad']/var_g['PctGood'])
        var_g.eval('IV = (PctBad - PctGood) * WOE / 100', inplace=True)

        # calculate F-score losses and cummulative bad rates
        var_g = poolingScores(var_g)
        var_g = var_g[['Class','LBound','UBound','Total','PctTotal','Bad','PctBad',
                       'BadRate','Good','PctGood','GoodRate','WOE','IV',
                       'F_standalone','F_combine','F_loss', 
                       'CumBad','CumTotal','CumBadRate']]
    else: 
        pass

    var_g.set_index('Class', drop=True, inplace=True)
    print('\nVar name:\t{}\nTotal steps:\t{}\nIV:\t{}'.format(var_name, 
                                                              count,
                                                              var_g.IV.sum()))
    
    return var_g
  

###############################################################################
def CatFastBin(raw_data, var_list, target):
        
    var_list.append(target)
    data = raw_data[var_list]
    
    # IV table
    iv_table = pd.DataFrame(columns=['var_name','info_val'])
    #iv_table = []
    # Bin Info table
    bin_info = pd.DataFrame(columns=['var_name', 'clus_num', 'category', 'WOE',
                                     'Bad', 'Good', 'PctTotal', 'Total', 
                                     'BadRate', 'IV'])
    #bin_info = []
    
    #initialize index number
    idx = 0
    
    for var_name in var_list[0:-1]: 
        # binning 
        var = data[[var_name, target]]
        var_g = ClassDiscrete(data=var, feature=var_name, 
                              target=target, ordered=True)
        var_g.set_index(var_name,inplace=True,drop=True)
        categories = var_g.index.values
        cutoffs = {}
        for i,v in enumerate(categories):
            cutoffs[v] = str(i+1)
        
        # variable IVs
        info_val = var_g[var_g.IV != np.inf].IV.sum()
        iv_table_new = pd.DataFrame({'info_val': info_val, 
                                     'var_name': var_name}, 
                                     index=[idx])
        iv_table = iv_table.append(iv_table_new)
        idx += 1
        
        # bin info
        bin_info_new = pd.DataFrame({'var_name': var_name, 
                                     'WOE': var_g.WOE.values, 
                                     'category': categories, 
                                     'clus_num': list(cutoffs.values()),
                                     'Bad': var_g.Bad.values,
                                     'Good': var_g.Good.values,
                                     'Total': var_g.Total.values,
                                     'PctTotal': var_g.PctTotal.values,
                                     'BadRate': var_g.BadRate.values,
                                     'IV': var_g.IV.values,
                                     })
        bin_info = bin_info.append(bin_info_new)
        
    # rearrange the columns
    iv_table = iv_table[['var_name', 'info_val']]
    bin_info = bin_info[['var_name', 'category', 'clus_num','Bad','Good',
                         'Total','PctTotal','BadRate', 'WOE','IV']]
    bin_info.reset_index(drop=True, inplace=True)
    iv_table.sort_values('info_val', ascending=False, inplace=True)
    
    return iv_table, bin_info
    

############################################################################### 
def stripbounds_l(bound):
    """
    Get the number on the lower bound
    from string '(lower, upper]'
    """
    
    s1 = bound.split(' ')[0]    
    try:
        ss1 = s1.strip('(').strip(',')
        l_bound = float(ss1)
    except ValueError:
        ss1 = s1.strip('[').strip(',')
        l_bound = float(ss1)
    return l_bound

###############################################################################    
def stripbounds_u(bound):
    """
    Get the number on the upper bound
    from string '(lower, upper]'
    """
    
    s2 = bound.split(' ')[1]
    ss2 = s2.strip(']')
    u_bound = float(ss2)
    return u_bound    


###############################################################################
def bin_info_table(var_g, var_name):
    bin_info = pd.DataFrame({'var_name': var_name, 
                             'WOE': var_g.WOE.values, 
                             'bucket': var_g.index.tolist(), 
                             'clus_num': var_g.reset_index().index.values.tolist(),
                             'LBound': None,
                             'UBound': None,
                             'Bad': var_g.Bad.values,
                             'Good': var_g.Good.values,
                             'Total': var_g.Total.values,
                             'PctTotal': var_g.PctTotal.values,
                             'BadRate': var_g.BadRate.values,
                             'IV': var_g.IV.values,
                             })
    
    bin_info['clus_num'] = bin_info['clus_num'].apply(lambda x: int(x)+1)
    bin_info['LBound'] = bin_info['bucket'].apply(stripbounds_l)
    bin_info['UBound'] = bin_info['bucket'].apply(stripbounds_u)
    
    bin_info = bin_info[['var_name','bucket','clus_num','LBound','UBound',
                         'Bad','Good','Total','PctTotal','BadRate','WOE','IV']]
    bin_info.reset_index(drop=True, inplace=True)

    return bin_info
        
###############################################################################
def NumFastBin(raw_data, var_list, target):
    
    var_list.append(target)
    data = raw_data[var_list]
    
    # IV table
    iv_table = pd.DataFrame(columns=['var_name','info_val'])
    #iv_table = []
    # Bin Info table
    bin_info = pd.DataFrame(columns=['var_name',
                                     'bucket',
                                     'clus_num',
                                     'LBound',
                                     'UBound',
                                     'WOE',
                                     'Bad',
                                     'Good',
                                     'PctTotal',
                                     'Total',
                                     'BadRate',
                                     'IV'])
    
    #initialize index number
    idx = 0
    
    for var_name in var_list[0:-1]: 
        # binning        
        var_g = NumAutoBin(data,var_name,target)
        
        # variable IVs
        info_val = var_g[var_g.IV != np.inf].IV.sum()
        iv_table_new = pd.DataFrame({'info_val': info_val,
                                     'var_name': var_name,
                                     },
                                     index=[idx])
        
        iv_table = iv_table.append(iv_table_new)
        idx += 1
        
        # bin info
        bin_info_new = pd.DataFrame({'var_name': var_name,
                                     'WOE': var_g.WOE.values, 
                                     'bucket': var_g.index.tolist(),
                                     'clus_num': var_g.reset_index().index.values.tolist(),
                                     'LBound': None,
                                     'UBound': None,
                                     'Bad': var_g.Bad.values,
                                     'Good': var_g.Good.values,
                                     'Total': var_g.Total.values,
                                     'PctTotal': var_g.PctTotal.values,
                                     'BadRate': var_g.BadRate.values,
                                     'IV': var_g.IV.values,
                                     })
        
        bin_info_new['clus_num'] = bin_info_new['clus_num'].apply(lambda x: int(x)+1)
        bin_info_new['bucket'] = bin_info_new['bucket'].apply(str)
        bin_info_new['LBound'] = bin_info_new['bucket'].apply(lambda x: stripbounds_l(x) \
                                                                if x!='NA' else x)
        bin_info_new['UBound'] = bin_info_new['bucket'].apply(lambda x: stripbounds_u(x) \
                                                                if x!='NA' else x)
        
        bin_info = bin_info.append(bin_info_new)
        
    # rearrange the columns
    iv_table = iv_table[['var_name','info_val']]
    bin_info = bin_info[['var_name',
                         'bucket',
                         'clus_num',
                         'LBound',
                         'UBound',
                         'Bad',
                         'Good',
                         'Total',
                         'PctTotal',
                         'BadRate',
                         'WOE',
                         'IV']]
                         
    bin_info.reset_index(drop=True, inplace=True)
    iv_table.sort_values('info_val', ascending=False, inplace=True)
    
    return iv_table, bin_info
    
###############################################################################    
def bin_info_tbl_c(var_g, var_name, cutoffs):
    """
    similar to bin_info_table function
    but for categorical variables
    in mannual binning process 
    """
    bin_info = pd.DataFrame({'var_name': var_name,                                                      
                             'category':  list(cutoffs.keys()), 
                             'clus_num': list(cutoffs.values()),
                             'Bad': 0,
                             'Good': 0,
                             'Total': 0,
                             'PctTotal': 0,
                             'BadRate': 0,
                             'WOE': 0, 
                             'IV': 0,
                               })
    bin_info = bin_info[['var_name', 'category', 'clus_num','Bad','Good','Total','PctTotal','BadRate', 'WOE','IV']]
    bin_info[['Bad' ,'Good', 'Total', 'PctTotal', 'BadRate', 'WOE', 'IV']] \
    = var_g.loc[bin_info.clus_num, ['Bad' ,'Good', 'Total', 'PctTotal', 'BadRate', 'WOE', 'IV']].values

    return bin_info


###################################################################################
def coarseContinuous(data,feature,target,cutoffs,bin_update,decision):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    data1 = data[[feature, target]]
    var_n, grouped_nan=nullClass(data1, feature, target)
  
    var_n['Class'] = pd.cut(var_n[feature], cutoffs, include_lowest=True)

    grouped = var_n.groupby('Class',
                            as_index=False)[target].agg({'Total':'count',
                                                         'Bad':'sum',
                                                         'BadRate':'mean'})
    bounds = var_n.groupby('Class',
                            as_index=False)[feature].agg({'LBound':'min',
                                                         'UBound':'max'})

    grouped['LBound'] = bounds['LBound']
    grouped['UBound'] = bounds['UBound']
    grouped.eval('Good = Total - Bad', inplace=True)
    grouped.eval('GoodRate = 1 - BadRate', inplace=True)
    grouped['PctTotal'] = grouped['Total'] / grouped['Total'].sum() * 100
    grouped['PctGood'] = grouped['Good'] / grouped['Good'].sum() * 100
    grouped['PctBad'] = grouped['Bad'] / grouped['Bad'].sum() * 100
    grouped['WOE'] = np.log(grouped['PctBad']/grouped['PctGood'])
    grouped.eval('IV = (PctBad - PctGood) * WOE / 100', inplace=True)

    var_g =  grouped[['Class','LBound','UBound','Total','PctTotal',
                        'Bad','PctBad','BadRate','Good','PctGood',
                        'GoodRate','WOE','IV']]
    
    var_g = var_g.append(grouped_nan)
    var_g['PctTotal'] = var_g['Total'] / var_g['Total'].sum() * 100
    var_g['PctGood'] = var_g['Good'] / var_g['Good'].sum() * 100
    var_g['PctBad'] = var_g['Bad'] / var_g['Bad'].sum() * 100
    var_g['WOE'] = np.log(var_g['PctBad']/var_g['PctGood'])
    var_g.eval('IV = (PctBad - PctGood) * WOE / 100', inplace=True)
    #var_g["lift"]=var_g["BadRate"]/(float(var_g.Bad.sum())/float(var_g.Total.sum()))
    var_g = var_g[['Class','LBound','UBound','Total','PctTotal','Bad','PctBad',
               'BadRate','Good','PctGood','GoodRate','WOE','IV']] 
    
    
    bin_info_new = pd.DataFrame({'var_name': feature,
                                     'WOE': var_g.WOE.values, 
                                     'bucket': var_g.Class.tolist(),
                                     'clus_num': var_g.reset_index().index.values.tolist(),
                                     'LBound': None,
                                     'UBound': None,
                                     'Bad': var_g.Bad.values,
                                     'Good': var_g.Good.values,
                                     'Total': var_g.Total.values,
                                     'PctTotal': var_g.PctTotal.values,
                                     'BadRate': var_g.BadRate.values,
                                     'IV': var_g.IV.values
                                     })
    
    bin_info_new['clus_num'] = bin_info_new['clus_num'].apply(lambda x: int(x)+1)
    bin_info_new['bucket'] = bin_info_new['bucket'].apply(str)
    
    bin_info_new['LBound'] = bin_info_new['bucket'].apply(lambda x: stripbounds_l(x) \
                                                        if x!='NA' else x)
    bin_info_new['UBound'] = bin_info_new['bucket'].apply(lambda x: stripbounds_u(x) \
                                                            if x!='NA' else x)
    bin_info_new = bin_info_new[['var_name',
                         'bucket',
                         'clus_num',
                         'LBound',
                         'UBound',
                         'Bad',
                         'Good',
                         'Total',
                         'PctTotal',
                         'BadRate',
                         'WOE',
                         'IV']]
    bin_info_new.set_index("var_name",inplace=True)
    if bin_update[bin_update.index==feature].shape[0]!=0:
        bin_update.drop(feature,inplace=True)
        
    if decision==1:
        bin_update=bin_update.append(bin_info_new)
    
    iv=bin_info_new.IV.sum()
    print iv
    return bin_info_new,bin_update,iv

################################################################################################
def coarseDiscrete(data,feature,target,mapping,bin_update,decision):
    pd.options.mode.chained_assignment = None  # default='warn'

    data = data[[feature, target]]
    data[feature] = data[feature].apply(str)

    if mapping != None:
        data['clus'] = data[feature].apply(lambda x: mapping[x])
        grouped = data.groupby(by='clus',
                               as_index=False)[target].agg({'Total':'count',
                                                            'Bad':'sum',
                                                            'BadRate':'mean'}) 
        grouped.set_index('clus',inplace=True,drop=True)
        categories = grouped.index.values
    else:
        grouped = data.groupby(by=feature,
                               as_index=False)[target].agg({'Total':'count',
                                                           'Bad':'sum',
                                                               'BadRate':'mean'})

        grouped.set_index(feature,inplace=True,drop=True)
        categories = grouped.index.values
    grouped.eval('Good = Total - Bad', inplace=True)
    grouped.eval('GoodRate = 1 - BadRate', inplace=True)
    grouped['PctTotal'] = grouped['Total'] / grouped['Total'].sum() * 100
    grouped['PctGood'] = grouped['Good'] / grouped['Good'].sum() * 100
    grouped['PctBad'] = grouped['Bad'] / grouped['Bad'].sum() * 100
    grouped['WOE'] = np.log(grouped['PctBad']/grouped['PctGood'])
    grouped.eval('IV = (PctBad - PctGood) * WOE / 100', inplace=True)
    #grouped=grouped.sort_values('BadRate')
    
    cutoffs = {}
    for i,v in enumerate(categories):
        cutoffs[v] = str(i+1)
    bin_info_new = pd.DataFrame({'var_name': feature, 
                                    'WOE': grouped.WOE.values, 
                                    'category': categories, 
                                    'clus_num': list(cutoffs.values()),
                                    'Bad': grouped.Bad.values,
                                    'Good': grouped.Good.values,
                                    'Total': grouped.Total.values,
                                    'PctTotal': grouped.PctTotal.values,
                                    'BadRate': grouped.BadRate.values,
                                    'IV': grouped.IV.values,
                                    })
    bin_info_new=bin_info_new[["var_name"	,"category"	,"clus_num",	"Bad",	"Good"	,'Total'	,"PctTotal"	,"BadRate"	,'WOE',	'IV']]
    bin_info_new.set_index("var_name",inplace=True,drop=True)
    if bin_update[bin_update.index==feature].shape[0]!=0:
        bin_update.drop(feature,inplace=True)
        
    if decision==1:
        bin_update=bin_update.append(bin_info_new)
    
    iv=bin_info_new.IV.sum()
    print iv
    return bin_info_new,bin_update,iv