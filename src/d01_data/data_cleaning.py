#!/usr/bin/env python
# coding: utf-8

# Required Packages

#DS Packages
import pandas as pd
import numpy as np

#Utils
from dateutil.relativedelta import relativedelta
from datetime import datetime
from datetime import timedelta

#ignore warnings
import warnings

#Custom Util funcs
import os
import sys
from pathlib import Path
module_path = str(Path.cwd().parents[0] / "d00_utils")
if module_path not in sys.path:
    sys.path.append(module_path)

#Import Utility Functions
from utility import print_full, make_lowercase, fix_spelling_mistakes

# #### Data Cleaning Functions
def cap_outliers(col, iqr_threshold=1.5, verbose=False, no_negative = False, manual_bounds = False, cap_only_lower=False):
    '''Caps outliers to closest existing value within threshold (IQR).'''
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    
    lbound = Q1 - iqr_threshold * IQR
    ubound = Q3 + iqr_threshold * IQR
    if no_negative:
        if lbound < 0:
            lbound = 0
        
    outliers = (col < lbound) | (col > ubound)
    if verbose:
        print(f'  Number of outliers:{len(outliers)}')
    
    col.loc[col < lbound] = col.loc[~outliers].min()
    if cap_only_lower:
        return col
    col.loc[col > ubound] = col.loc[~outliers].max()

    if verbose:
            print('\n'.join(
                ['Capping outliers by the IQR method:',
                 f'   IQR threshold: {iqr_threshold}',
                 f'   Lower bound: {lbound}',
                 f'   Upper bound: {ubound}\n']))

    return col

def get_category_level_diffs(series_arr):
    
    series_arr = make_lowercase(series_arr)
    
    series_1_cats =  series_arr[0].cat.categories
    series_2_cats =  series_arr[1].cat.categories 
    

    diffs = list(set(series_1_cats).symmetric_difference(set(series_2_cats)))
    print(f'Number of different Category Levels between datasets: {len(diffs)}')
    return diffs

def unify_cat_levels(series_arr):
    
    series_arr = make_lowercase(series_arr)
    diffs = get_category_level_diffs(series_arr)
    
    
    for i in range(len(series_arr)):
        series_arr[i] = ['diff' if x in diffs else x for x in series_arr[i]]
        print(f'Series {i+1}: values unified')
        print(f'Number of rows with diff category level: {series_arr[i].count("diff")}')
    
    return series_arr[0], series_arr[1] 

def changeDtypes(_df):
    # date_time variables
    date_vars = ['order_date',
                 'delivery_date',
                 'user_dob',
                 'user_reg_date']
    #category variables
    category_vars = ['item_size',
                     'item_color',
                     'user_title',
                     'user_state',
                     'brand_id',
                     'item_id',
                     'user_id']
    # numeric variables
    numeric_vars = 'item_price'
    
    # Convert to appropriate variable dtype
    _df[date_vars] = _df[date_vars].astype('datetime64[ns]')
    _df[category_vars] = _df[category_vars].astype('category')
    _df[numeric_vars] = _df[numeric_vars].astype(np.float32)
    return _df

def impute_delivery_time(_df, verbose = False):
    print(f'Imputing Delivery times for: {_df.name}')
    # create df with only nonNA values of order date and delivery date
    df_order_delivery = _df[['order_date', 'delivery_date']].dropna()
    
    #use timedelta to calculate difference between order and delivery date
    df_delivery_time = df_order_delivery.apply(
        lambda x: (x['delivery_date'] - x['order_date']).days,
        axis = 1,
        result_type='expand')
    
    # compoute mean_deliver_time
    mean_deliver_time = df_delivery_time.median()
    
    if verbose:
        print('Mean Delivery Time is: ',mean_deliver_time)
        
    #add median delivery_time to order date of missing values
    _df['delivery_date'] =_df.apply(
        lambda x: x['order_date'] + timedelta(days=mean_deliver_time) \
        if pd.isnull(x['delivery_date']) \
        else x['delivery_date'], axis=1, result_type='expand')
    if verbose:
        print('No. of Missing Values after imputation:',_df['delivery_date'].isnull().sum())
    return _df

def impute_na_user_dob(col):
    # get normalized distribution 
    vc = col.value_counts(normalize=True)
    # fill na values wih randomized values following the distribution
    return np.random.choice(vc.index, p=vc.values, size=col.isna().sum()) 

if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    #Random Seed Constant
    random_seed = 420

    #set numpy random seed 
    np.random.seed(random_seed)

    #Known Data Set
    df = pd.read_csv('../../data/01_raw/BADS_WS2021_known.csv')
    df.name = 'Known Data'

    #Unknown Data Set
    df_unknown = pd.read_csv('../../data/01_raw/BADS_WS2021_unknown.csv')
    df_unknown.name = 'Unknown Data'

    #add 100000 to each order_item_id 
    df['order_item_id'] = df['order_item_id']
    df['order_item_id'] = df['order_item_id']+ 100000
    #set order_item_id as index
    df.set_index('order_item_id', inplace = True)
    df_unknown.set_index('order_item_id',inplace = True)

    # change dtypes for both df's
    df = changeDtypes(df)
    df_unknown = changeDtypes(df_unknown)

    #iterate through all entries and add 22 years to all entries that are lower than 2016
    df['delivery_date'] = df['delivery_date'].apply(
        lambda x: x + relativedelta(years=22) 
        if x.year < 2016 \
        else x)
    df_unknown['delivery_date'] = df_unknown['delivery_date'].apply(
        lambda x: x + relativedelta(years=22)  
        if x.year < 2016 \
        else x )



    # Now we can find the **median** `delivery_time` and add this to the `order_date` of the missing entries in order to impute the missing `delivery_date`. We use the median since it should be more robust than the *mean*.
    df = impute_delivery_time(df, verbose = True)
    df_unknown = impute_delivery_time(df_unknown, verbose = True)

    # ### 2.2 Imputing and cleaning of  `user_dob`

    # get normalized distribution of all user_dob that are higher than 1920
    vc = df['user_dob'][df['user_dob'].dt.year > 1920].value_counts(normalize=True)
    # insert random value following the distribution
    df['user_dob'] = df['user_dob'].apply(lambda x: np.random.choice(vc.index, p=vc.values) if x.year < 1920 else x)


    # We can now impute the missing values using random values following the distribution.


    df.loc[df['user_dob'].isna(), 'user_dob'] = impute_na_user_dob(df['user_dob'])


    df_unknown.loc[df_unknown['user_dob'].isna(), 'user_dob'] = impute_na_user_dob(df_unknown['user_dob'])

    # ## 3. Outliers and Invalid Values

    # ### 3.2 Categorical Data

    # #### 3.2.1 `item_color`

    # dict of spelling mistakes to correct
    colors_spelling_mistakes = {
    'brwn' : 'brown', 
    'blau' : 'blue',
    '?': 'undefined' 
    }
    #fix spelling mistakes for both datasets
    df['item_color'] = fix_spelling_mistakes(df['item_color'], colors_spelling_mistakes)
    df_unknown['item_color'] = fix_spelling_mistakes(df_unknown['item_color'],colors_spelling_mistakes)

    df['item_color'], df_unknown['item_color'] = unify_cat_levels([df['item_color'], df_unknown['item_color']])


    # #### 3.2.2 `item_size`
    df['item_size'], df_unknown['item_size'] = unify_cat_levels([df['item_size'], df_unknown['item_size']])

    # Somewhere Dtypes have been changed..
    df = changeDtypes(df)
    df_unknown = changeDtypes(df_unknown)
    
    # ## 4. Export Cleaned Data 
    df.to_pickle('../../data/02_intermediate/BADS_WS2021_known_cleaned.pkl')
    df_unknown.to_pickle('../../data/02_intermediate/BADS_WS2021_unknown_cleaned.pkl')





