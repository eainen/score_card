
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import datetime

# define map function for numerical variables
def raw2group(raw, bin_tbl):
    
    lbound = bin_tbl.LBound.values
    ubound = bin_tbl.UBound.values
    tbl_idx = 0
    
    for i in range(len(ubound)):

        # NA handler 
        if pd.isnull(raw) and pd.isnull(lbound[i]):
            tbl_idx = i 
        else: 
            if ~pd.isnull(raw) and ~pd.isnull(lbound[i]) and raw<=ubound[i] and raw>lbound[i]:
                tbl_idx = i
            else: continue
    return  bin_tbl.loc[tbl_idx, 'clus_num']

def stability_stamp_c(raw_data,timestamp,var_name,target,mapping=None):
   
    var_ts = raw_data[[timestamp, var_name,target]]
    var_ts.set_index(timestamp, drop=True, inplace=True)

    if mapping==None:
        map_category={}
        for i,v in enumerate(sorted(var_ts[var_name].unique())):
            map_category[v] = str(i+1)
        var_ts['group'] = var_ts[var_name].apply(str).map(map_category)
    else:
        var_ts['group'] = var_ts[var_name].apply(str).map(mapping)
    var_ts.drop(var_name,axis=1,inplace=True)
    var_ts = pd.get_dummies(columns=['group'], data=var_ts)
    var_ts_wk=var_ts.groupby(timestamp).mean()


    var_ts_wk.loc[:, 'group_1':].plot.area(ylim=[0,1], 
                                alpha=0.5,
                                title='PctTotal',
                                #legend=None,
                               )
    group_list = var_ts.loc[:,"group_1":].columns.tolist()
    for group_num in group_list:
        var_ts[group_num + '_b'] = var_ts[group_num] * var_ts[target]
    bin_info_cnt=var_ts.groupby(timestamp).sum()
    var_ts_wk_br = var_ts.groupby(timestamp).mean()
    group_list2 = var_ts_wk_br.loc[:, 'group_1_b':].columns.tolist()
    for i, name in enumerate(group_list2):
        var_ts_wk_br.eval(' {} = {} / {} '.format(name+'r', name, group_list[i]), inplace=True)

    var_ts_wk_br.loc[:, 'group_1_br':].plot(title='BadRate',
                                        ylim=[0.0, 0.5],
                                      #legend=False,
                                       ) 
    tabel1=var_ts_wk.T
    psi_tabel=tabel1.loc["group_1":,:]
    psi_res=pd.DataFrame()
    for i in psi_tabel.columns:
        psi_res["psi_"+"%s"%i]=(psi_tabel.iloc[:,1]-psi_tabel.loc[:,i])*np.log(psi_tabel.iloc[:,1]/psi_tabel.loc[:,i])
        psi=psi_res.sum()
    print psi
    for name in group_list2:
        var_ts_wk_br.drop(name,axis=1,inplace=True)
    bin_info_pct=var_ts_wk_br.copy()
    return bin_info_cnt,bin_info_pct,psi
    
def stability_stamp_n(raw_data,timestamp,var_name,target,bin_info):

    var_ts = raw_data[[timestamp, var_name,target]]
    var_ts.set_index(timestamp, drop=True, inplace=True)
    var_g=bin_info.copy()
    var_g.reset_index(inplace=True)
    if len(var_g[var_g.bucket=="NA"].index)>0:
        var_g.loc[var_g[var_g.bucket=="NA"].index,["bucket","LBound","UBound"]]=np.nan
    else: 
        pass

    var_ts['group'] = var_ts[var_name].apply(lambda x: raw2group(x, var_g))
   
    var_ts = pd.get_dummies(columns=['group'], data=var_ts)
    
    var_ts_wk=var_ts.groupby(timestamp).mean()
  
 
    var_ts_wk.loc[:, 'group_1':].plot.area(ylim=[0,1], 
                                alpha=0.5,
                                title='PctTotal',
                                #legend=None,
                               )

        
    group_list = var_ts.loc[:,"group_1":].columns.tolist()
    for group_num in group_list:
        var_ts[group_num + '_b'] = var_ts[group_num] * var_ts[target]
    bin_info_cnt=var_ts.groupby(timestamp).sum()
    var_ts_wk_br = var_ts.groupby(timestamp).mean()
    group_list2 = var_ts_wk_br.loc[:, 'group_1_b':].columns.tolist()
    for i, name in enumerate(group_list2):
        var_ts_wk_br.eval(' {} = {} / {} '.format(name+'r', name, group_list[i]), inplace=True)
    
    var_ts_wk_br.loc[:, 'group_1_br':].plot(title='BadRate',
                                        ylim=[0.0, 0.5],
                                      #legend=False,
                                       ) 
    tabel1=var_ts_wk.T
    psi_tabel=tabel1.loc["group_1":,:]
    psi_res=pd.DataFrame()
    for i in psi_tabel.columns:
        psi_res["psi_"+"%s"%i]=(psi_tabel.iloc[:,1]-psi_tabel.loc[:,i])*np.log(psi_tabel.iloc[:,1]/psi_tabel.loc[:,i])
        psi=psi_res.sum()
    print psi
    for name in group_list2:
        var_ts_wk_br.drop(name,axis=1,inplace=True)
    var_ts_wk_br.drop(var_name,axis=1,inplace=True)
    bin_info_cnt.drop(var_name,axis=1,inplace=True)
    bin_info_pct=var_ts_wk_br.copy()
    return bin_info_cnt,bin_info_pct,psi


def stability_month_n(raw_data,timestamp,var_name,target,bin_info):

    var_ts = raw_data[[timestamp, var_name,target]]
    var_ts[timestamp] = var_ts[timestamp].apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m%d'))
    var_ts.set_index(timestamp, drop=True, inplace=True)
    var_g=bin_info.copy()
    var_g.reset_index(inplace=True)
    if len(var_g[var_g.bucket=="NA"].index)>0:
        var_g.loc[var_g[var_g.bucket=="NA"].index,["bucket","LBound","UBound"]]=np.nan
    else: 
        pass
    
    var_ts['group'] = var_ts[var_name].apply(lambda x: raw2group(x, var_g))
    var_ts = pd.get_dummies(columns=['group'], data=var_ts)
    
    var_ts_wk=var_ts.resample('M').mean()
    
    
    var_ts_wk.loc[:, 'group_1':].plot.area(ylim=[0,1], 
                                alpha=0.5,
                                title='PctTotal',
                                #legend=None,
                               )

    
        
    group_list = var_ts.loc[:,"group_1":].tolist()
    for group_num in group_list:
        var_ts[group_num + '_b'] = var_ts[group_num] * var_ts[target]
    var_ts_wk_br = var_ts.resample('M').mean()
    group_list2 = var_ts_wk_br.loc[:, 'group_1_b':].columns.tolist()
    for i, name in enumerate(group_list2):
        var_ts_wk_br.eval(' {} = {} / {} '.format(name+'r', name, group_list[i]), inplace=True)
    
    var_ts_wk_br.loc[:, 'group_1_br':].plot(title='BadRate',
                                        ylim=[0.0, 0.5],
                                    #legend=False,
                                    ) 
    tabel1=var_ts_wk.T
    psi_tabel=tabel1.loc["group_1":,:]
    psi_res=pd.DataFrame()
    for i in psi_tabel.columns:
        psi_res["psi_"+"%s"%i]=(psi_tabel.iloc[:,2]-psi_tabel.loc[:,i])*np.log(psi_tabel.iloc[:,2]/psi_tabel.loc[:,i])
        psi=psi_res.sum()
    print psi
    return var_ts_wk_br,psi