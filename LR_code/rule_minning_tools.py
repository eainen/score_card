
def dum_x(data_input,keys):
    # unstack
    ## keys: unstack keys
    data_input.set_index(keys, inplace=True)
    data_input = data_input[~data_input.index.duplicated(keep=False)]  
    data_input = data_input.unstack()
    data_input.columns = data_input.columns.droplevel()
    return data_input


def normalization(data_mat):
    count=0
    for column in data_mat.columns:
    #   print column
        if column=='label' or ('pct' in column) or ('ratio' in column): 
            print column
            continue
#         data_mat[column] = data_mat[column].apply(lambda x: to_number(x))
        mean_v = data_mat[column].mean()
        std_v = data_mat[column].std()
        max_value = mean_v+4*std_v
        
    #   print column,max_value
        data_mat.loc[data_mat[column] > max_value,column] = max_value
        data_mat[column] = data_mat[column] / max_value
        count += 1
        print count
    return data_mat


def fillna(var_list,reference,data):
    count=0
    for feature in var_list:
        if feature==reference : 
            print 'unneeded'
            continue
        data[feature] = data[feature][data[reference].notnull()].fillna(0)
        count +=1
        print count
    return data


