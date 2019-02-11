#! /usr/bin/env python2.7
# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np

# define map function for numerical variables

def raw2woe(raw, bin_tbl):
    
    lbound = bin_tbl.LBound.values
    ubound = bin_tbl.UBound.values
    tbl_idx = 0
    bin_tbl.reset_index(inplace=True,drop=True)
    for i in range(len(ubound)):

        # NA handler 
        if pd.isnull(raw) and pd.isnull(lbound[i]):
            tbl_idx = i 
        else: 
            if ~pd.isnull(raw) and ~pd.isnull(lbound[i]) and raw<=ubound[i] and raw>lbound[i]:
                tbl_idx = i
            else: continue
    return  bin_tbl.loc[tbl_idx, 'WOE']

def woeTransform_n(data, var_list,bin_tbl,key1=None,key2=None,key3=None):
    # key1=None,key2=None,key3=None used to add other useful columns to woe_data
    woe_data = pd.DataFrame()
    bin_info=bin_tbl.copy()
    #bin_info.set_index("var_name",inplace=True,drop=True)
    i = 1
    for var in var_list:
        bin_tbl = bin_info[bin_info.var_name==var]
        #bin_tbl.reset_index(drop=True, inplace=True)
        if bin_tbl[bin_tbl.bucket=="NA"].shape[0]>0:
            bin_tbl.loc[bin_tbl[bin_tbl.bucket=="NA"].index,["bucket","LBound","UBound"]]=np.nan
        else: 
            pass
        woe_data[var] = data[var].apply(lambda x: raw2woe(x,bin_tbl))
            
        print('Transformed: %s \t %d/%d' % (var, i, len(var_list)))
      
        i += 1
    if key1:
        woe_data[key1]=data[key1]
    if key2:
        woe_data[key2]=data[key2]
    if key3:
        woe_data[key3]=data[key3]
    return woe_data

def woeTransform_c(data,var_list,bin_tbl,mappings,key1=None,key2=None,key3=None):
     # key1=None,key2=None,key3=None used to add other useful columns to woe_data
    woe_data = pd.DataFrame()
    bin_info=bin_tbl.copy()
    bin_info.set_index("var_name",inplace=True,drop=True)
    i = 1
    for var_name in var_list:
        mapping=mappings[var_name]
        var_g=bin_info.loc[var_name,:]
        woe_data[var_name] = data[var_name].apply(str).map(mapping)
        map_series = pd.Series(var_g.WOE.values, index=var_g.category.astype(str))
        woe_data[var_name] = woe_data[var_name].apply(str).map(map_series)

        print('Transformed: %s \t %d/%d' % (var_name, i, len(var_list)))

        i += 1
    if key1:
        woe_data[key1]=data[key1]
    if key2:
        woe_data[key2]=data[key2]
    if key3:
        woe_data[key3]=data[key3]
    return woe_data




    
