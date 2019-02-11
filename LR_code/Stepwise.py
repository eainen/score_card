#!bash
# -*- coding:utf-8 -*-
#author Jason
#date 2017/2/27
#describe 逐步回归实现

import statsmodels.formula.api as smf
import pandas as pd
import  os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
data_path = root_dir + "/traindata/train.csv"

class Stepwise(object):

    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    def forward_selected(self,data, response):
        remaining = set(data.columns)
        remaining.remove(response)
        selected = []
        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
                score = smf.ols(formula, data).fit().rsquared_adj
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        formula = "{} ~ {}".format(response, ' + '.join(selected))
        model = smf.ols(formula, data).fit()
        return model

if __name__=="__main__":
    SP = Stepwise()
    # url = "http://data.princeton.edu/wws509/datasets/salary.dat"
    # df = pd.read_csv(url, sep='\\s+')
    df = pd.read_csv(data_path)
    data = df[df.columns[1:]]
    model = SP.forward_selected(data, 'y_value')
    print model.model.formula
    print model.rsquared_adj

