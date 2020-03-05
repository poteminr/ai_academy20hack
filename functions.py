import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score


def REN(X):
    return X.rename(columns={"client_dk":'client_id'})

def prep_date(data):
    data["trans_month"] = ((data["trans_date"] // 30) % 12 + 1)
    data["trans_year"] = ((data["trans_date"] // 365) + 1)
     
def get_first_last_transaction_date(data):
    ret = data.groupby('client_id')["trans_date"].agg(["min", "max"])
    ret.columns = ["first_transaction_date", "last_transaction_date"]
    return ret

def get_minmax_yearly_trans(data):
    ret = data.groupby(['client_id','trans_year'])["amount_rur"].agg(['count', 'sum'])\
        .groupby(['client_id'])[['count', 'sum']].agg(['max', 'min']).reset_index()

    ret.columns = ['client_id', 'yeartrans_max', 'yeartrans_min', 'yearcount_max', 'yearcount_min']
    ret["yearcount_diff"] = ret["yearcount_max"] - ret["yearcount_min"]
    return ret

def get_minmax_monthly_trans(data):
    ret = data.groupby(['client_id','trans_month'])["amount_rur"].agg(['count', 'sum'])\
        .groupby(['client_id'])[['count', 'sum']].agg(['max', 'min']).reset_index()

    ret.columns = ['client_id', 'monthtrans_max', 'monthtrans_min', 'monthcount_max', 'monthcount_min']
    ret["monthcount_diff"] = ret["monthcount_max"] - ret["monthcount_min"]
    return ret

def get_stat(data, func='count'):
    if func == "count":
        counter_df = data.groupby(['client_id','small_group'])['amount_rur'].count()
        cat_counts = counter_df.reset_index().pivot(index='client_id', \
                                                      columns='small_group', values='amount_rur')
    
        cat_counts = cat_counts.fillna(0)
        cat_counts.columns = ['small_group_'+str(i) for i in cat_counts.columns]
    
    elif func == "sum":
        counter_df = data.groupby(['client_id','small_group'])['amount_rur'].sum()
        cat_counts = counter_df.reset_index().pivot(index='client_id', \
                                                      columns='small_group', values='amount_rur')
    
        cat_counts = cat_counts.fillna(0)
        cat_counts.columns = ['small_group_sum_'+str(i) for i in cat_counts.columns]
    
    elif func == 'std':
        counter_df = data.groupby(['client_id','small_group'])['amount_rur'].std()
        cat_counts = counter_df.reset_index().pivot(index='client_id', \
                                                      columns='small_group', values='amount_rur')
    
        cat_counts = cat_counts.fillna(0)
        cat_counts.columns = ['small_group_std_'+str(i) for i in cat_counts.columns]
        
    elif func == "median":
        counter_df = data.groupby(['client_id','small_group'])['amount_rur'].median()
        cat_counts = counter_df.reset_index().pivot(index='client_id', \
                                                      columns='small_group', values='amount_rur')
    
        cat_counts = cat_counts.fillna(0)
        cat_counts.columns = ['small_group_median_'+str(i) for i in cat_counts.columns]
        
    return cat_counts

def get_row_with_max_col_val(df, col, group_col):
    idx = df.groupby(group_col)[col].transform(max) == df[col]
    return df[idx]

def preproc_data(transactions, is_train=True, y_train = None, test_id=None):
    
    prep_date(transactions)
    get_minmax_yearly_trans(transactions)
    get_minmax_monthly_trans(transactions)
    
    
    agg_features=transactions.groupby('client_id')['amount_rur'].agg(['sum','mean','std','min','max']).reset_index()

    max_group = get_row_with_max_col_val(transactions, 'amount_rur', 'client_id')[ ["client_id", "small_group"]]
    print(max_group.shape)

    counter_df_train = transactions.groupby(['client_id','small_group'])['amount_rur'].count()
    
    sum_df_train = transactions.groupby(['client_id','small_group'])['amount_rur'].agg(["sum","max"])
    
    
    
    minmax_dayly_transactions_train = \
    transactions.groupby(['client_id','trans_date'])["amount_rur"].agg(['count', 'sum'])\
    .groupby(['client_id'])[['count', 'sum']].agg(['max', 'min']).reset_index()

    minmax_dayly_transactions_train.columns = ['client_id', 'daytrans_max', 'daytrans_min', 'daycount_max', 'daycount_min']
    
    
    
    cat_counts_train=counter_df_train.reset_index().pivot(index='client_id', \
                                                      columns='small_group',values='amount_rur')
    cat_counts_train=cat_counts_train.fillna(0)
    cat_counts_train.columns=['sg_count_'+str(i) for i in cat_counts_train.columns]
    
    
    
    cat_sum_train = sum_df_train.reset_index().pivot(index='client_id', \
                                                      columns='small_group',values=["sum","max"])
    cat_sum_train = cat_sum_train.fillna(0)
    cat_sum_train.columns=['sg_sum_'+str(i) for i in cat_sum_train.columns]
    
    
    if is_train:
        train=pd.merge(y_train,agg_features,on='client_id')
    
    else:
        train = pd.merge(test_id, cat_counts_train.reset_index(),on='client_id')
        print(train.shape)
        
    train = pd.merge(train, cat_counts_train.reset_index(),on='client_id')
    print(train.shape)


    
    train=pd.merge(train, cat_sum_train.reset_index(),on='client_id')
    print(train.shape)


    train = pd.merge(train, max_group.reset_index()[["client_id", "small_group"]],on='client_id')
    print(train.shape)
    
    
    train=pd.merge(train, minmax_dayly_transactions_train,on='client_id')
    
    
    train = pd.merge(train,get_minmax_monthly_trans(transactions),on='client_id')
    print(train.shape)


    train = pd.merge(train,get_minmax_yearly_trans(transactions),on='client_id')
    print(train.shape)


    
    train = pd.merge(train,get_first_last_transaction_date(transactions).reset_index(),on='client_id')
    print(train.shape)


    
    
    CAT = pd.concat([get_stat(transactions, func='count'), get_stat(transactions, func='sum'),
                        get_stat(transactions, func='std'), get_stat(transactions, func='median')],axis=1)
    
    train = pd.merge(train, CAT.reset_index(), on='client_id')
    
    return train.drop_duplicates(subset="client_id")
    
def average_score(model, columns, train_data, y_train):
    scores = []
    for col in columns:
        score = cross_val_score(
            model,
            train_data,
            y_train[col],
            cv = 4,
            scoring = "roc_auc",
            verbose = 1,
        )
        print(col, score, np.mean(score))
        scores.append( np.mean(score) )
    return np.mean(scores)