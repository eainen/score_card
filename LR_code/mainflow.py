# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:21:53 2017

@author: jue.wang01
"""
#%% 
import os
os.chdir("D:\\Work Docs\\评分卡\\现金贷评分卡\\现金贷筛选评分卡V2_201704\\BQstandalone")

#%% Basic tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 1 - Import data
# import from csv
data_train = pd.read_csv("data\\version_2\\data_train_v2.csv")
data_test = pd.read_csv("data\\version_2\\data_test_v2.csv")
data_validate = pd.read_csv("data\\version_2\\data_validate_v2.csv")

# or, import from previously pickled
df_train = pd.read_pickle("data\\train.pkl") # this has no keys

# exclude IDs 
df_train = data_train.iloc[:, 13:]
df_test = data_test.iloc[:, 13:]
df_validate = data_validate.iloc[:, 13:]

# add target column
df_train['TARGET'] = data_train['TARGET']
df_test['TARGET'] = data_test['TARGET']
df_validate['TARGET'] = data_validate['TARGET']

# pickle interims
df_train.to_pickle("data\\version_2\\train.pkl")
df_test.to_pickle("data\\version_2\\test.pkl")
df_validate.to_pickle("data\\version_2\\validate.pkl")


#%% 2 - Feature types, numercial or categorical

# read variable dictionary file
var_dict = pd.read_excel("data\\version_2\\var_dict.xlsx")

var_list_n = var_dict[var_dict.TYPE == 'NUM'].VAR_NAME.tolist() #numerical
var_list_c = var_dict[var_dict.TYPE == 'CAT'].VAR_NAME.tolist() #categorical

#%% 3 - Auto/Fast binning

# fire up the toolkit
os.chdir("D:\\Work Docs\\评分卡\\现金贷评分卡\\现金贷筛选评分卡V2_201704\\BQstandalone")
import PoolingAlgo as pal

# fast bin categorical variables
iv_c, bin_c = pal.CatFastBin(df_train, var_list_c, 'TARGET')

# fast bin numerical variables
iv_n, bin_n = pal.NumFastBin(df_train, var_list_n, 'TARGET')

# output binning results
with pd.ExcelWriter("results\\autobinning.xlsx") as writer:
    iv_c.to_excel(writer, sheet_name='IV_cat')
    bin_c.to_excel(writer, sheet_name='BIN_cat')
    iv_n.to_excel(writer, sheet_name='IV_num')
    bin_n.to_excel(writer, sheet_name='BIN_num')

#%% 4 - Coarse classing (further pooling and mannual adjusting)

"""
# pick variables with accepatble level of information value
# do it in Jupyter notebook environment for better visuals
# output same format as with auto binning  
"""


#%% 5 - Transform to WOE
    
# load data and binning results
data = pd.read_pickle("data\\train.pkl")

# remove anomaly observations
data = data[-data.CUS_SEX.isnull()]
data = data[-data.OTHER_PERSON_TYPE.isnull()]

bin_info_c = pd.read_excel("results\\coarsebinning_c_cand.xlsx", sheetname='BIN')
bin_info_n = pd.read_excel("results\\coarsebinning_n_cand.xlsx", sheetname='BIN')

"""
# the updated var_dict_cand.xlsx shall have a column specifying
# whether to variable is to be included for modeling
"""
var_dict = pd.read_excel("results\\var_dict_cand.xlsx")
var_dict = var_dict[var_dict.MODEL == 1]
var_dict = var_dict[(var_dict.TYPE=='NUM')|(var_dict.TYPE=='CAT')]

# convert raw data to WOE
# call function in woeTransform.py
from woeTransform import woeTransform
woe_data = woeTransform(data=data, 
                        bin_info_c=bin_info_c, 
                        bin_info_n=bin_info_n, 
                        var_dict=var_dict)

# store interims
woe_data.to_pickle("data\\train_woe.pkl")

#%% 6 - Feature selection
# correlations
corr = woe_data.corr(method='pearson')
corr.to_excel("results\\correlation_woe_pearson.xlsx")

corr = woe_data.corr(method='spearman')
corr.to_excel("results\\correlation_woe_spearman.xlsx")

#%% 7 - Stepwise regression

X = woe_data.drop('TARGET', axis=1)
y = woe_data['TARGET']

initial_candidates = []                    
potential_candidates = []
for cand in X.columns.tolist():
    if cand not in initial_candidates:
        potential_candidates.append(cand)
    else: continue

potential_candidates.remove('WOE_DELAY_TIMES_V2')
#potential_candidates.remove('WOE_AVG_DAYS_V2')
potential_candidates.remove('WOE_EVER_M1_TIMES_V2')
potential_candidates.remove('WOE_SIX_DELAY_TIMES_V2')
potential_candidates.remove('WOE_HIS_DELAYDAYS_V2')
potential_candidates.remove('WOE_M1_DAYS_V2')
#potential_candidates.remove('WOE_DD_DIFF')
#potential_candidates.remove('WOE_DD_FAIL')
potential_candidates.remove('WOE_MAX_DPD_V2')
potential_candidates.remove('WOE_MAX_CPD_V2')
potential_candidates.remove('WOE_PAY_DELAY')

#potential_candidates.remove('WOE_DIFF_PPL')
potential_candidates.remove('WOE_OTHER_PERSON_TYPE')
#potential_candidates.remove('WOE_CUS_SEX')
potential_candidates.remove('WOE_PERIODS')
#potential_candidates.remove('WOE_ONTIME_PAY')

# call from StepwiseLR.py
model = fwd_select(X, y, 
                   initial_candidates, 
                   potential_candidates, 
                   max_n_feature=13)
res = model.fit()
print(res.summary())

# call from PerformanceMeasure.py
X['Intercept'] = 1
prob = scorebucket(X, y, model, res)
_, bucket = ksdistance(prob)
X.drop('Intercept', axis=1, inplace=True)

# save model files
os.chdir("D:\\Work Docs\\评分卡\\现金贷评分卡\\现金贷筛选评分卡V2_201704\\BQstandalone")
import pickle
openfile1 = open('results\\model_v1\\lr_model_v1.pkl','wb')
pickle.dump(model, openfile1)
openfile2 = open('results\\model_v1\\lr_res_v1.pkl','wb')
pickle.dump(res, openfile2)



#%% call SAS
import saspy 
sas = saspy.SASsession()
data_sas = sas.dataframe2sasdata(df=woe_data, table="saspytest", libref="WORK")

stat = sas.sasstat()

params_name = list(res.params.index)
params_name.remove('WOE_PAY_DELAY')
params_name.remove('WOE_ONTIME_PAY')

model =  """
TARGET(event='1') = {}
/selection = stepwise
slentry = 0.05
slstay = 0.05
""".format(" ".join(params_name[1:]))
         
lr_model = stat.logistic(data=data_sas, model=model)

print(lr_model.PARAMETERESTIMATES)

lr_model.PARAMETERESTIMATES.to_excel("sas_res_13.xlsx")


# statsmodel single-step regression
import statsmodels.formula.api as smf
params_name.remove('Intercept')
formula = "{} ~ {} + 1".format('TARGET',' + '.join(params_name))
lr_model = smf.logit(formula, woe_data)
lr_res = lr_model.fit()
print(lr_res.summary())

X['Intercept'] = 1
prob = scorebucket(X, y, lr_model, lr_res)
_, bucket = ksdistance(prob)
X.drop('Intercept', axis=1, inplace=True)
#%% 8 - Alternative models
