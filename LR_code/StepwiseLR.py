# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:54:24 2017

@author: Daniel
"""

#%% import libraries
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
                      
#%% forward selection function


def fwd_select(X_train,y_train,initial_candidates,potential_candidates,tol=0.05,max_n_feature=200):
    """
    Forward selection by AIC .
    
    > Input:
    >> X_train - without the Intercept column
    >> y_train - with the target column only
    >> initial_candidates - as list of feature names to start with
                            can be an empty list
    >> potential_candidates - as list of feature names to select from
    >> tol - the significance level (mainly for elimination step)
    
    > Output:
    >> model (statsmodels object)
        
    Referrence:
    http://statsmodels.sourceforge.net/stable/generated/
    statsmodels.discrete.discrete_model.Logit.html
    
    > Update on 2017/4/25:
        add an eliminating condition in each steps
        using a P-value threshold (default=0.05)
        
    
    """
    # assign response column name
    response = 'TARGET'
    # training set including target variable
    data_train = X_train.copy()
    data_train[response] = y_train
    
    # Set initial score
    current_formula = "{} ~ {} + 1".format(response, 
                                           ' + '.join(initial_candidates))
    current_score = smf.logit(current_formula, data_train).fit().aic
        
    # loop while the candidate number is 1 more than the initial number
    count = len(potential_candidates)
    score_list = []
    selected_candidates = initial_candidates
    
    
    for candidate in potential_candidates:
        formula = "{} ~ {} + 1".format(response,
                                        ' + '.join(selected_candidates\
                                                  +[candidate]))
        
        try:
             res = smf.logit(formula, data_train).fit(method='newton',
                                                      maxiter=100,
                                                      disp=0,
                                                      tol=tol, 
                                                      )
        except Exception as error:
            print "Skipped\t {} due to {}".format(candidate, error)
            potential_candidates.remove(candidate)
            count -= 1     
            continue 
        
        score = res.aic 
        converged = res.mle_retvals['converged']
        if converged:
            score_list.append((score,candidate))
        else: 
            print "Skipped\t {} due to not converged".format(candidate)
            continue
        
        
        score_list.sort() # ascending sort
        best_score, best_candidate = score_list[0] # lowest score
    
        if best_score < current_score:
        
            selected_candidates.append(best_candidate) #add to selecteds
            potential_candidates.remove(best_candidate) #remove from potentials
            current_score = best_score
            current_formula = "{} ~ {} + 1".format(response,
                                            ' + '.join(selected_candidates))
            print "Added\t {}".format(best_candidate)
        
        
        # backward elimination:
        # eliminating (P > tol) variable
            bk_res = smf.logit(current_formula, data_train).fit(method='newton',
                                                            maxiter=100,
                                                            disp=0,
                                                            tol=tol,)            
            p_values = bk_res.pvalues
            p_over = p_values[p_values > tol]
            if p_over.shape[0] > 0:                
                for name in p_over.index:
                    try:
                        selected_candidates.remove(name)
                        print "Removed\t {} due to PValue={}".format(name,p_over[name])
                    except ValueError:
                        continue
            else: 
                pass 
        else: 
            pass
    
    # too many variables
        if len(selected_candidates)>=max_n_feature:
            break
    
    # next loop
        count -= 1
        print count
        print "Left-overs: {}".format(len(potential_candidates))
        print "Selected: {}".format(len(selected_candidates))
        
    final_formula = "{} ~ {} + 1".format(response,
                                        ' + '.join(selected_candidates))
    final_model = smf.logit(final_formula, data_train)
    
    if len(potential_candidates) > 0:
        print "-" * 50
        print "Left-overs: \n-- {}".format('\n-- '.join(potential_candidates))
    
    return final_model,final_formula     





def bkwd_select(X_train, y_train, threshold=5.0):
    """
    Backward selection by VIF
    
    Input:
        X_train - training set exogenous variables
        y_train - training set endogenous variable
        threshold - the level of VIF indicating high multicollinearity
    Output:
        model
    """
    response = 'TARGET'
    
    data_train = X_train.copy()
    data_train[response] = y_train
    
    namelist = X_train.columns
    try:
        namelist = namelist.drop('Intercept')
    except ValueError:
        print("Intercept not in data columns")
    
    removed = pd.Series()
    
    while True:
        vif = pd.Series([variance_inflation_factor(X_train[namelist].values,idx)\
                          for idx in range(X_train[namelist].shape[1])],
                        index=namelist, name='vif')
        if vif.max() > threshold:
            removed = removed.append(vif[vif == vif.max()])
            print("Removed {}".format(vif[vif == vif.max()]))
            selected = vif[vif < vif.max()]
                           
        else:
            selected = vif
        
        if selected.index.tolist() == namelist.tolist():
            break
        else: namelist = selected.index 
    
    formula = "{} ~ {} + 1".format(response, ' + '.join(namelist))
    model = smf.logit(formula, data_train)
    
    return model


#%% Stepwise 


#%% For test drive
'''
from sklearn.cross_validation import train_test_split

#%% load test data
data = pd.read_pickle('data_woe_transformed.pkl')

X = data.drop(['CONTRACT_NO','TARGET'], axis=1)
y = data['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=100)

initial_candidates = ['WOE_MAX_CPD','WOE_NOCALL_DAY']
potential_candidates = ['WOE_CUS_SEX','WOE_ONTIME_PAY','WOE_CUS_EDUCATION',
                        'WOE_APP_COUNT']

model = fwd_select(X_train,y_train,initial_candidates,potential_candidates)
print(model.fit().summary())
                        
#%% Try out with a perfectly corelated feature
X_train2 = X_train.copy()
X_train2.eval('WOE_MAX_CPD_2 = -WOE_MAX_CPD * 0.8', inplace=True)

initial_candidates = ['WOE_MAX_CPD','WOE_NOCALL_DAY']
potential_candidates = ['WOE_MAX_CPD_2','WOE_CUS_SEX','WOE_APP_COUNT']

model = fwd_select(X_train2,y_train,initial_candidates,potential_candidates)
print(model.fit().summary())

'''   
