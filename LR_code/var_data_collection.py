#! /usr/bin/env python2.7
# -*- coding:utf-8 -*-

'''
created on 20180408
used for sjd_fqd_model
'''

import pandas as pd 
import numpy as np

def get_hive_data(customer_name):
    print 'get_hive_data'
    #### load sample data
    from sqlalchemy.engine import create_engine
    hive_engine = create_engine('hive://wuzx@192.168.254.107:10000',
                                connect_args={'auth':'KERBEROS','kerberos_service_name':'hive',
                                'configuration': {'mapred.job.queue.name':  'root.anti_fraud'}})
    
    sql_1='''
        select * from anti_fraud.y_customer_feature_input_m where customer_name='{customer_name}'
    '''.format(customer_name=customer_name)
    sample_keys=pd.read_sql(sql_1,hive_engine)
    
    sql_2='''
        select distinct t1.input_key                                        
                ,t2.outside_name as feature                               
                ,t1.value                                              
                ,t1.recall_date                                        
                ,t1.source                                             
                ,t1.type
        from anti_fraud.factor_train_%s_v2_r t1
        join anti_fraud.factors_name_mapping t2
        on t1.feature=t2.internal_name
        where t2.version='V2'
    '''%customer_name
    sample_v2=pd.read_sql(sql_2,hive_engine)
    sample_v2_shape=sample_v2[~sample_v2[["input_key","recall_date"]].duplicated()].shape[0]
    var_dict_v2=sample_v2[~sample_v2[["feature","type"]].duplicated()][["feature","type"]]
    print 'sample_keys shape ',sample_keys.shape,' sample_v2 sample size ',sample_v2_shape
    return sample_keys,sample_v2,var_dict_v2

def clean_v2_data(sample_v2,var_dict_v2,fillna=True):
    print '===========================unstack_v2_data=============================='
    sample_v2_1=sample_v2[~sample_v2[["input_key","recall_date","feature"]].duplicated()][["input_key","recall_date","feature","value"]]
    sample_v2_2=sample_v2_1.set_index(["input_key","recall_date","feature"])["value"].unstack()
    sample_v2_2.reset_index(inplace=True)
    print 'input_v2 shape',sample_v2_2.shape
    print '=============================set data type==============================='
    sample_v2_2.iloc[:,2:]=sample_v2_2.iloc[:,2:].astype(np.float64)
    sample_v2_2.loc[:,"input_key"]=sample_v2_2.loc[:,"input_key"].apply(lambda x : unicode(str(x)))
    input_v2=sample_v2_2.copy()
    print 'input_v2 shape ',input_v2.shape
    if fillna==True:
        print '================================fill NA================================='
        feature_0M_1M=var_dict_v2[var_dict_v2.type=="app_0M_1M"].feature.tolist()
        feature_1M_2M=var_dict_v2[var_dict_v2.type=="app_1M_2M"].feature.tolist()
        feature_2M_3M=var_dict_v2[var_dict_v2.type=="app_2M_3M"].feature.tolist()
        print 'app var number ',len(feature_0M_1M),len(feature_1M_2M),len(feature_2M_3M)
        fillna_dict={"DEVICE_Stability_0M_1M":feature_0M_1M,
                    "DEVICE_Stability_1M_2M": feature_1M_2M,
                    "DEVICE_Stability_2M_3M":feature_2M_3M}
        for reference in ["DEVICE_Stability_0M_1M","DEVICE_Stability_1M_2M","DEVICE_Stability_2M_3M"]:
            var_list=fillna_dict[reference]
            count=0
            print reference
            for feature in var_list:
                if feature==reference : 
                    print 'unneeded'
                    continue
                input_v2.loc[:,feature] = input_v2[feature][input_v2[reference].notnull()].fillna(0)
                count +=1
                if count % 100 == 0:
                    print count
        print input_v2[feature_0M_1M[300:304]].isnull().sum()
        print input_v2[feature_1M_2M[296:300]].isnull().sum()
        print input_v2[feature_2M_3M[100:104]].isnull().sum()
    else:
        pass
    return input_v2

def get_hbase_data(sample_keys,var_list=None):
    print 'get_hbase_data'
    import happybase
    hbase_ips = ['192.168.254.67' ,'192.168.254.68', '192.168.255.66']
    connection = happybase.Connection(hbase_ips[0], timeout=30000)
    table = connection.table('bt_iaudience')
    res = {}
    n = 0
    abnormal_key=sample_keys[sample_keys.imei.apply(lambda x: (len(x)>16 or len(x)<10))].input_key.tolist()
    print 'abnormal key',len(abnormal_key)
    if len(abnormal_key)>0:
        sample_keys=sample_keys[~sample_keys.input_key.isin(abnormal_key)]
    if var_list==None:
        for imei in sample_keys.imei:
            res[str(imei)] = table.row(b'{imei}'.format(imei=imei), columns=['A:CID_JID', 'A:CPL_HHM_CHILD_HC', 'A:CPL_INDM_GEND_S', 'A:CPL_INDM_MARRC2', 'A:CPL_INDM_NATI', 'A:CPL_INDM_AGE_C5', 'A:CPL_HHM_CHILD_CHLI', 'A:CID_MODEL', 'A:CPL_DVM_BRAD', 'A:CPL_DVM_HF', 'A:CPL_DVM_ISP', 'A:CPL_DVM_OS', 'A:CPL_DVM_PUPR', 'A:CPL_DVM_RESO', 'A:CPL_DVM_SCSIZE', 'A:CPL_DVM_TIME', 'A:CPL_DVM_TYPE', 'A:CPL_INDM_VEIC_VEID', 'A:FIM_FISM_CONL_CIR', 'A:FIM_FISM_INCL', 'A:GBM_BHM_PURB_CONP', 'A:GBM_BHM_PURB_PREF', 'A:SOM_OCM_CAREER', 'A:GBM_HBM_S', 'A:GBM_BHM_APPP_APPR_S', 'A:GBM_BHM_PURB_PURW', 'A:GBM_BHM_PURB_SUPR', 'A:GBM_BHM_REAB_REAP', 'A:APP_HOBY_BUS', 'A:APP_HOBY_TICKET', 'A:APP_HOBY_TRAIN', 'A:APP_HOBY_FLIGHT', 'A:APP_HOBY_TAXI', 'A:APP_HOBY_SPECIAL_DRIVE', 'A:APP_HOBY_HIGH_BUS', 'A:APP_HOBY_OTHER_DRIVE', 'A:APP_HOBY_RENT_CAR', 'A:APP_HOBY_STARS_HOTEL', 'A:APP_HOBY_YOUNG_HOTEL', 'A:APP_HOBY_HOME_HOTEL', 'A:APP_HOBY_CONVERT_HOTEL', 'A:APP_HOBY_BANK_UNIN', 'A:APP_HOBY_ALIPAY', 'A:APP_HOBY_THIRD_PAY', 'A:APP_HOBY_INTERNET_BANK', 'A:APP_HOBY_FOREIGN_BANK', 'A:APP_HOBY_MIDDLE_BANK', 'A:APP_HOBY_CREDIT_CARD', 'A:APP_HOBY_CITY_BANK', 'A:APP_HOBY_STATE_BANK', 'A:APP_HOBY_FUTURES', 'A:APP_HOBY_VIRTUAL_CURRENCY', 'A:APP_HOBY_FOREX', 'A:APP_HOBY_NOBLE_METAL', 'A:APP_HOBY_FUND', 'A:APP_HOBY_COLLECTION', 'A:APP_HOBY_STOCK', 'A:APP_HOBY_ZONGHELICAI', 'A:APP_HOBY_CAR_LOAN', 'A:APP_HOBY_DIVIDE_LOAN', 'A:APP_HOBY_STUDENT_LOAN', 'A:APP_HOBY_CREDIT_CARD_LOAN', 'A:APP_HOBY_CASH_LOAN', 'A:APP_HOBY_HOUSE_LOAN', 'A:APP_HOBY_P2P', 'A:APP_HOBY_LOAN_PLATFORM', 'A:APP_HOBY_SPORT_LOTTERY', 'A:APP_HOBY_WELFARE_LOTTERY', 'A:APP_HOBY_DOUBLE_BALL', 'A:APP_HOBY_LOTTERY', 'A:APP_HOBY_FOOTBALL_LOTTERY', 'A:APP_HOBY_MARK_SIX', 'A:APP_HOBY_WECHAT', 'A:APP_HOBY_SUMMARY_LIVE', 'A:APP_HOBY_SHORT_VIDEO', 'A:APP_HOBY_SOCIAL_LIVE', 'A:APP_HOBY_TRAVEL_LIVE', 'A:APP_HOBY_SUMMARY_VIDEO', 'A:APP_HOBY_SPORTS_VIDEO', 'A:APP_HOBY_GAME_LIVE', 'A:APP_HOBY_BEAUTY_LIVE', 'A:APP_HOBY_COS_LIVE', 'A:APP_HOBY_SELF_PHOTO', 'A:APP_HOBY_TV_LIVE', 'A:APP_HOBY_CULTURE_LIVE', 'A:APP_HOBY_SHOW_LIVE', 'A:APP_HOBY_EDU_LIVE', 'A:APP_HOBY_SPORTS_LIVE', 'A:APP_HOBY_STARS_LIVE', 'A:APP_HOBY_READ_LISTEN', 'A:APP_HOBY_SUNMMARY_NEWS', 'A:APP_HOBY_WOMEN_HEL_BOOK', 'A:APP_HOBY_ARMY_NEWS', 'A:APP_HOBY_CARTON_BOOK', 'A:APP_HOBY_PHY_NEWS', 'A:APP_HOBY_FAMOUSE_BOOK', 'A:APP_HOBY_FINCAL_NEWS', 'A:APP_HOBY_FINCAL_BOOK', 'A:APP_HOBY_FUN_NEWS', 'A:APP_HOBY_EDU_MED', 'A:APP_HOBY_KONGFU', 'A:APP_HOBY_TECH_NEWS', 'A:APP_HOBY_LOOK_FOR_MED', 'A:APP_HOBY_ENCOURAGE_BOOK', 'A:APP_HOBY_CAR_INFO_NEWS', 'A:APP_HOBY_HUMERIOUS', 'A:APP_HOBY_CARDS_GAME', 'A:APP_HOBY_SPEED_GAME', 'A:APP_HOBY_ROLE_GAME', 'A:APP_HOBY_NET_GAME', 'A:APP_HOBY_RELAX_GAME', 'A:APP_HOBY_KONGFU_GAME', 'A:APP_HOBY_GAME_VIDEO', 'A:APP_HOBY_TALE_GAME', 'A:APP_HOBY_DIAMONDS_GAME', 'A:APP_HOBY_TRAGEDY_GAME', 'A:APP_HOBY_OUTDOOR', 'A:APP_HOBY_MOVIE', 'A:APP_HOBY_CARTON', 'A:APP_HOBY_BEAUTIFUL', 'A:APP_HOBY_LOSE_WEIGHT', 'A:APP_HOBY_PHY_BOOK', 'A:APP_HOBY_FRESH_SHOPPING', 'A:APP_HOBY_WIFI', 'A:APP_HOBY_CAR_PRO', 'A:APP_HOBY_LIFE_PAY', 'A:APP_HOBY_PET_MARKET', 'A:APP_HOBY_OUT_FOOD', 'A:APP_HOBY_FOOD', 'A:APP_HOBY_PALM_MARKET', 'A:APP_HOBY_WOMEN_HEAL', 'A:APP_HOBY_RECORD', 'A:APP_HOBY_CONCEIVE', 'A:APP_HOBY_SHARE', 'A:APP_HOBY_COOK_BOOK', 'A:APP_HOBY_BUY_RENT_HOUSE', 'A:APP_HOBY_CHINESE_MEDICINE', 'A:APP_HOBY_JOB', 'A:APP_HOBY_HOME_SERVICE', 'A:APP_HOBY_KRAYOK', 'A:APP_HOBY_FAST_SEND', 'A:APP_HOBY_PEOPLE_RESOUSE', 'A:APP_HOBY_MAMA_SOCIAL', 'A:APP_HOBY_GAY_SOCIAL', 'A:APP_HOBY_HOT_SOCIAL', 'A:APP_HOBY_MARRY_SOCIAL', 'A:APP_HOBY_CAMPUS_SOCIAL', 'A:APP_HOBY_LOVERS_SOCIAL', 'A:APP_HOBY_ECY', 'A:APP_HOBY_STRANGER_SOCIAL', 'A:APP_HOBY_ANONYMOUS_SOCIAL', 'A:APP_HOBY_CITY_SOCIAL', 'A:APP_HOBY_FANS', 'A:APP_HOBY_FIN', 'A:APP_HOBY_MIDDLE', 'A:APP_HOBY_IT', 'A:APP_HOBY_PRIMARY', 'A:APP_HOBY_BABY', 'A:APP_HOBY_ONLINE_STUDY', 'A:APP_HOBY_FOREIGN', 'A:APP_HOBY_DRIVE', 'A:APP_HOBY_SERVANTS', 'A:APP_HOBY_CHILD_EDU', 'A:APP_HOBY_UNIVERSITY', 'A:APP_HOBY_CAR_SHOPPING', 'A:APP_HOBY_SECONDHAND_SHOPPING', 'A:APP_HOBY_ZONGHE_SHOPPING', 'A:APP_HOBY_PAYBACK', 'A:APP_HOBY_DISCOUNT_MARKET', 'A:APP_HOBY_BABY_SHOPPING', 'A:APP_HOBY_WOMEN_SHOPPING', 'A:APP_HOBY_REBATE_SHOPPING', 'A:APP_HOBY_GROUP_BUY', 'A:APP_HOBY_GLOBAL_SHOPPING', 'A:APP_HOBY_SHOPPING_GUIDE', 'A:APP_HOBY_SEX_SHOPPING', 'A:APP_HOBY_SMOTE_OFFICE'])
            n += 1
            if n % 5000 == 0:
                print(str(n) + " complete!")
    else:
        for imei in sample_keys.imei:
            res[str(imei)] = table.row(b'{imei}'.format(imei=imei), columns=var_list)
            n += 1
            if n % 5000 == 0:
                print(str(n) + " complete!")
                      
    print 'input imei ',sample_keys.shape[0],' return bt sample ',len(res)
    sample_bt=pd.DataFrame(res).T
    new_columns=[]
    for var in sample_bt.columns:
        new_columns=new_columns+[var[2:]]
    sample_bt.columns=new_columns
    sample_bt.reset_index(inplace=True)
    sample_bt=sample_bt.rename(columns={"index":"imei"})
    sample_bt=sample_keys[["input_key","imei","recall_date","label"]].merge(sample_bt,left_on="imei",right_on="imei")
    print 'sample_bt shape ',sample_bt.shape
    return sample_bt

def clean_bt_data(sample_bt):
    print 'clean_bt_data'
    import iaudience_a_cleaning as clean
    clean_dict=pd.read_csv("C:\\Users\\jiguang\\Desktop\\LRCode\\LRCode_from_wuzx\\LRCode\\label_clean_type.csv")
    input_bt,var_dict_bt=clean.clean_iaudience_a(sample_bt,clean_dict)
    input_bt.loc[:,"input_key"]=input_bt.loc[:,"input_key"].apply(lambda x : unicode(str(x)))
    var_dict_bt[~var_dict_bt.var_name.isin(["input_key","recall_date","label"])]
    print 'input_bt shape ',input_bt.shape
    return input_bt,var_dict_bt

def get_vars_together(var_dict_v2,var_dict_bt,input_bt,input_v2):
    print 'get_vars_together'
    var_dict_v2.columns=["feature","type"]
    var_dict_v2["class"]='N'
    var_dict_bt.columns=["feature","class","var_name_bt"]
    var_dict=var_dict_v2.append(var_dict_bt)
    input_bt["bt_mark"]=1
    input_v2["v2_mark"]=1
    input_data=input_bt.merge(input_v2,left_on=["input_key","recall_date"],right_on=["input_key","recall_date"],how="outer")
    print 'input_data shape ',input_data.shape
    return var_dict,input_data
    







