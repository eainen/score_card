#! /usr/bin/env python2.7
# -*- coding:utf-8 -*-

import pandas as pd 
import numpy as np


def dif_month(list):
    x=str(list[0])
    y=str(list[1])
    if len(y)==4:
        dif=(int(x[0:4])-int(y))*12+int(x[4:6])-1
    elif len(y)>=6:
        dif=(int(x[0:4])-int(y[0:4]))*12+int(x[4:6])-int(y[4:6])
    else :
        dif=np.nan
    return dif 

def clean_iaudience_a(iaudience_data,clean_dict):

    var=iaudience_data.columns.tolist()

    var_c=clean_dict[clean_dict.type=='char'].var_name.tolist()
    var_c=list(set(var_c).intersection(set(var)))
    len_c=len(var_c)
    print len_c,"character variables need to be cleaned"

    var_n=clean_dict[clean_dict.type=='num'].var_name.tolist()
    var_n=list(set(var_n).intersection(set(var)))
    len_n=len(var_n)
    print len_n,"numeric variables need to be cleaned"

    var_clean=clean_dict[clean_dict.type=='clean'].var_name.tolist()
    var_clean=list(set(var_clean).intersection(set(var)))
    len_clean=len(var_clean)
    print len_clean,"variables need to be dealt"

    cleaned_iaudience_data=pd.DataFrame()
    cleaned_iaudience_data=iaudience_data[["input_key","recall_date","label"]]
    ## clean var_clean
    count=0
    for var in var_clean:
        if var=='GBM_BHM_PURB_PURW':
            tmp=pd.DataFrame()
            tmp["x"]=iaudience_data[var]
            tmp["mark"]=iaudience_data[var].apply(lambda x:len(str(x).split("&")))
            ###split
            ttt=tmp[tmp.mark==0]
            for i in range(1,6):
                tt=tmp[tmp.mark==i][["x","mark"]]
                for j in range(1,i+1):
                    tt[j]=tt["x"].apply(lambda x: str(x).split("&")[j-1])
                locals()['tmp'+'%s' %i]=tt.copy()   
                ttt=ttt.append(locals()['tmp'+'%s' %i])
            ### unstack
            t=ttt.iloc[:,:5]
            t=t.unstack(level=0)
            t=pd.DataFrame(t)
            t.reset_index(inplace=True)
            t.columns=["feature","imei","value"]
            t["value1"]=t["value"].apply(lambda x:str(x).split(",")[0])
            t["value2"]=t["value"].apply(lambda x:str(x).split(",")[-1])
            t=t[["imei","value1","value2"]]
            t=t[t["value1"]!="nan"]
            t0=pd.DataFrame(t.groupby("imei",as_index=True)["value1"].count())
            t0.columns=["hit_num"]
            t.set_index(["imei","value1"],inplace=True,drop=True)
            t=t.unstack()
            t.columns = t.columns.droplevel()  
            ### left join
            tmp=tmp.merge(t,left_index=True,right_index=True,how="left")
            tmp=tmp.merge(t0,left_index=True,right_index=True,how="left")
            tmp["团购"]=tmp["团购"].astype(np.float64)
            cleaned_iaudience_data["TG__"+var+"_N"]=tmp["团购"]
            cleaned_iaudience_data["DG__"+var+"_N"]=tmp["导购"]
            cleaned_iaudience_data["BJ__"+var+"_N"]=tmp["比价"]
            cleaned_iaudience_data["WL__"+var+"_N"]=tmp["网络商城"]
            cleaned_iaudience_data["GW__"+var+"_N"]=tmp["购物分享"]
            cleaned_iaudience_data["HTN_"+var+"_C"]=tmp["hit_num"]
        if var=='CPL_DVM_RESO':
            tmp=pd.DataFrame()
            tmp["n"]=iaudience_data[var].apply(lambda x: len(str(x).split("*")))
            tmp["x"]=iaudience_data[var].apply(lambda x: str(x).split("*")[0])
            tmp["y"]=iaudience_data[var].apply(lambda x: str(x).split("*")[-1])
            index=tmp[tmp.n!=2].index
            tmp.loc[index,["x","y"]]=np.nan
            tmp["x"]=tmp["x"].astype(np.float64)
            tmp["y"]=tmp["y"].astype(np.float64)
            cleaned_iaudience_data["ONL_"+var+"_N"]=tmp["x"]*tmp["y"]
        if var=='CPL_INDM_AGE_C5':
            tmp=pd.DataFrame()
            tmp["mark"]=iaudience_data[var].apply(lambda x:len(str(x).split("&")))
            index=tmp[tmp["mark"]!=2].index
            tmp["x"]=iaudience_data[var].apply(lambda x: str(x).split("&")[0])
            tmp["y"]=iaudience_data[var].apply(lambda x: str(x).split("&")[-1])
            tmp.loc[index,["x","y"]]=np.nan
            tmp["x"]=tmp["x"].astype(np.float64)
            tmp["y"]=tmp["y"].astype(np.float64)
            cleaned_iaudience_data["AG1_"+var+"_C"]=tmp["x"]
            cleaned_iaudience_data["CON_"+var+"_N"]=tmp["y"]
        if var=='CPL_DVM_TIME':
            tmp=pd.DataFrame()
            tmp["x"]=iaudience_data[var]
            tmp["y"]=iaudience_data["recall_date"]
            tmp["z"]=tmp.x.apply(lambda x: str(x).replace("年","").replace("月","").replace("日","")[0:6])
            tmp["p"]=tmp[["y","z"]].apply(lambda row : dif_month(row),axis=1)
            tmp["mark"]=tmp["x"].apply(lambda x:len(str(x)))
            index=tmp[tmp["mark"]<=4].index
            tmp.loc[index,["p"]]=np.nan
            cleaned_iaudience_data["ONL_"+var+"_N"]=tmp["p"]
        if var=='FIM_FISM_INCL':
            tmp=pd.DataFrame()
            tmp["mark"]=iaudience_data[var].apply(lambda x:len(str(x).split("&")))
            index=tmp[tmp["mark"]!=2].index
            tmp["x"]=iaudience_data[var].apply(lambda x: str(x).split("&")[0])
            tmp["y"]=iaudience_data[var].apply(lambda x: str(x).split("&")[-1])
            tmp.loc[index,["x","y"]]=np.nan
            tmp["y"]=tmp["y"].astype(np.float64)
            cleaned_iaudience_data["HML_"+var+"_C"]=tmp["x"]
            cleaned_iaudience_data["CON_"+var+"_N"]=tmp["y"]
        if var=='FIM_FISM_CONL_CIR':
            tmp=pd.DataFrame()
            tmp["mark"]=iaudience_data[var].apply(lambda x:len(str(x).split("&")))
            index=tmp[tmp["mark"]!=2].index
            tmp["x"]=iaudience_data[var].apply(lambda x: str(x).split("&")[0])
            tmp["y"]=iaudience_data[var].apply(lambda x: str(x).split("&")[-1])
            tmp.loc[index,["x","y"]]=np.nan
            tmp["y"]=tmp["y"].astype(np.float64)
            cleaned_iaudience_data["HML_"+var+"_C"]=tmp["x"]
            cleaned_iaudience_data["CON_"+var+"_N"]=tmp["y"]
        if var=='GBM_BHM_PURB_PREF':
            tmp=pd.DataFrame()
            tmp["x"]=iaudience_data[var]
            tmp["mark"]=iaudience_data[var].apply(lambda x:len(str(x).split("&")))
            ###split
            ttt=tmp[tmp.mark==0]
            for i in range(1,6):
                tt=tmp[tmp.mark==i][["x","mark"]]
                for j in range(1,i+1):
                    tt[j]=tt["x"].apply(lambda x: str(x).split("&")[j-1])
                locals()['tmp'+'%s' %i]=tt.copy()   
                ttt=ttt.append(locals()['tmp'+'%s' %i])
            ### unstack
            t=ttt.iloc[:,:5]
            t=t.unstack(level=0)
            t=pd.DataFrame(t)
            t.reset_index(inplace=True)
            t.columns=["feature","imei","value"]
            t["value1"]=t["value"].apply(lambda x:str(x).split(",")[0])
            t["value2"]=t["value"].apply(lambda x:str(x).split(",")[-1])
            t=t[["imei","value1","value2"]]
            t=t[t["value1"]!="nan"]
            t0=pd.DataFrame(t.groupby("imei",as_index=True)["value1"].count())
            t0.columns=["hit_num"]
            t.set_index(["imei","value1"],inplace=True,drop=True)
            t=t.unstack()
            t.columns = t.columns.droplevel()  
            ### left join
            tmp=tmp.merge(t,left_index=True,right_index=True,how="left")
            tmp=tmp.merge(t0,left_index=True,right_index=True,how="left")
            tmp["大众"]=tmp["大众"].astype(np.float64)
            cleaned_iaudience_data["DZ__"+var+"_N"]=np.where(tmp["大众"]==0,np.nan,tmp["大众"])
            cleaned_iaudience_data["CY__"+var+"_N"]=np.where(tmp["创意"]==0,np.nan,tmp["创意"])
            cleaned_iaudience_data["SC__"+var+"_N"]=np.where(tmp["奢侈"]==0,np.nan,tmp["奢侈"])
            cleaned_iaudience_data["SL__"+var+"_N"]=np.where(tmp["潮流"]==0,np.nan,tmp["潮流"])
            cleaned_iaudience_data["GD__"+var+"_N"]=np.where(tmp["高端"]==0,np.nan,tmp["高端"])
        if var=="CPL_DVM_SCSIZE":
            tmp=pd.DataFrame()
            tmp["x"]=iaudience_data[var].apply(lambda x:str(x).strip("英寸"))
            tmp["mark"]=iaudience_data[var].apply(lambda x: str(x)[-2:])
            index=tmp[tmp["mark"]!='\xaf\xb8'].index
            tmp.loc[index,["x"]]=np.nan
            tmp["x"]=tmp["x"].astype(np.float64)
            cleaned_iaudience_data["ONL_"+var+"_N"]=tmp["x"]
        if var=="CPL_INDM_GEND_S":
            tmp=pd.DataFrame()
            tmp["mark"]=iaudience_data[var].apply(lambda x:len(str(x).split("&")))
            index=tmp[tmp["mark"]!=2].index
            tmp["x"]=iaudience_data[var].apply(lambda x: str(x).split("&")[0])
            tmp["y"]=iaudience_data[var].apply(lambda x: str(x).split("&")[-1])
            tmp.loc[index,["x","y"]]=np.nan
            tmp["y"]=tmp["y"].astype(np.float64)
            tmp.loc[index,["x","y"]]=np.nan
            cleaned_iaudience_data["SEX_"+var+"_C"]=tmp["x"]
            cleaned_iaudience_data["CON_"+var+"_N"]=tmp["y"]
        count +=1
        print "clean",count,var

    count=0
    for var in var_c:
        cleaned_iaudience_data["ONL_"+var+"_C"]=iaudience_data[var]
        count +=1
        print "char",count,var

    count=0
    for var in var_n:
        tmp=pd.DataFrame()
        tmp[var]=iaudience_data[var]
        tmp["mark"]=tmp[var].apply(lambda x : len(str(x).split("&")))
        index=tmp[tmp["mark"]!=2].index
        tmp["x"]=tmp[var].apply(lambda x : str(x).split("&")[-1])
        tmp.loc[index,"x"]=np.nan
        tmp["x"]=tmp["x"].astype(np.float64)
        cleaned_iaudience_data["ONL_"+var+"_N"]=tmp["x"]
        count +=1
        print "num",count,var

    var_dict=pd.DataFrame(cleaned_iaudience_data.columns)
    var_dict.columns=["var_name"]
    var_dict["type"]=var_dict["var_name"].apply(lambda x: str(x).split("_")[-1])
    var_dict["var_name_en"]=var_dict["var_name"].apply(lambda x: str(x)[4:-2])

    for var in var_dict[var_dict.type=="C"].var_name.tolist():
        cleaned_iaudience_data[var]=cleaned_iaudience_data[var].astype(np.string_)
        cleaned_iaudience_data[var]=np.where(cleaned_iaudience_data[var]=='',np.nan,cleaned_iaudience_data[var])

    for var in var_dict[var_dict.type=="N"].var_name.tolist():
    	cleaned_iaudience_data[var]=cleaned_iaudience_data[var].astype(np.float)

    
    return cleaned_iaudience_data,var_dict

