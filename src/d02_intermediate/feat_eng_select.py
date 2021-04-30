
# Required Packages
#DS Packages
import pandas as pd
import numpy as np

#Utils
from dateutil.relativedelta import relativedelta
from datetime import datetime

import json

from optbinning import OptimalBinning

#ignore warnings
import warnings

#Custom Imports
import os
import sys
from pathlib import Path
paths = [str(Path.cwd().parents[0] / "d00_utils"), str(Path.cwd().parents[0] / "d01_data")]
for module_path in paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

# Import custom utility functions
from utility import print_full, make_lowercase, fix_spelling_mistakes
    
# Import custom Data Cleaning functions 
from data_cleaning import cap_outliers, unify_cat_levels, get_category_level_diffs


# Feature Engineering and Selection Functions
def calc_week_days(_df):
    weekDays = ("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
    _df['order_weekday'] = _df['order_date'].map(datetime.weekday)
    _df['order_weekday'] = _df['order_weekday'].apply(lambda x: weekDays[x])
    _df['order_weekday'] = _df['order_weekday'].astype('category')
    return _df

def calc_delivery_days(_df):
    delivery_days = _df.apply(lambda x: relativedelta(x['delivery_date'], x['order_date']).days, axis = 1)
    delivery_days = cap_outliers(delivery_days,verbose=True)
    delivery_days = delivery_days.astype(int)
    return delivery_days

def calc_user_account_age(_df):
    user_account_age_ser = _df.apply(lambda x: (x['order_date'] - x['user_reg_date']).days, axis = 1)
    user_account_age_ser = cap_outliers(user_account_age_ser, verbose=True)
    user_account_age_ser = user_account_age_ser.astype(int)
    return user_account_age_ser

def calc_user_age(_df):
    # claculate age based on time of order
    user_age =_df.apply(lambda x: (x['order_date'] - x['user_dob']).days/365, axis = 1)
    user_age = user_age.astype(int)
    user_age = cap_outliers(user_age,verbose=True)
    return user_age

def calc_number_of_total_orders_by_user(_df, user_dfs):
    total_orders_by_user =_df.apply(
        lambda row: user_dfs[row['user_id']]['order_date'].where(
            user_dfs[row['user_id']]['order_date'] > row['order_date']).count() , axis = 1)
    total_orders_by_user = cap_outliers(total_orders_by_user)
    return total_orders_by_user

def calc_has_bought_item_before(_df):
    has_bought_item_before = _df.apply(
        lambda row: len(_df[(_df['user_id'] == row['user_id']) \
                           & (_df['item_id'] == row['item_id']) \
                           & (_df['order_date']<row['order_date'])])!= 0
                            , axis=1)
    has_bought_item_before = has_bought_item_before.astype(int)
    return has_bought_item_before

def calc_is_first_purchase(_df):
    number_of_purchases = _df.apply(
        lambda row: len(_df[(_df['user_id'] == row['user_id']) \
                         & (_df['order_date']< row['order_date'])]) == 0
                        , axis=1)
    number_of_purchases = number_of_purchases.astype(int)
    return number_of_purchases 

def calc_number_of_items_in_order(_df):
    number_of_items_in_order = _df.apply(lambda row: len(_df[(_df['user_id'] == row['user_id']) & (_df['order_date'] == row['order_date'])]), axis = 1)
    return number_of_items_in_order

def calc_ordered_item_multiple_times_in_order(_df):
    ordered_item_multiple_times_in_order = _df.apply(
        lambda row: len(_df[(_df['order_date'] == row['order_date']) \
                            &(_df['user_id'] == row['user_id'])\
                            & (_df['item_id'] == row['item_id']) ]) > 1,
                            axis = 1)
    ordered_item_multiple_times_in_order = ordered_item_multiple_times_in_order.astype(int)
    return ordered_item_multiple_times_in_order

def optimal_binning(col, y ):
    optb = OptimalBinning(dtype='categorical', solver='cp', max_n_prebins = 80)
    optb.fit(col.values, y.values)
    return optb


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    #Random Seed Constant
    random_seed = 420

    #set numpy random seed 
    np.random.seed(random_seed)


    df = pd.read_pickle('../../data/02_intermediate/BADS_WS2021_known_cleaned.pkl')
    df_unknown = pd.read_pickle('../../data/02_intermediate/BADS_WS2021_unknown_cleaned.pkl')

    #Week Days
    df = calc_week_days(df)
    df_unknown = calc_week_days(df_unknown)


    # ### 2.2 Days till delivery
    df['delivery_days'] = calc_delivery_days(df)
    df_unknown['delivery_days'] = calc_delivery_days(df_unknown)


    # ### 2.3 User specific Features
    user_known_dfs = {}
    user_unknown_dfs = {}
    for user_id in sorted(df['user_id'].unique()):
        user_df =  df.loc[df['user_id'] == user_id]
        user_known_dfs[user_id] = user_df
    for user_id in sorted(df_unknown['user_id'].unique()):
        user_df =  df_unknown.loc[df_unknown['user_id'] == user_id]
        user_unknown_dfs[user_id] = user_df


    # #### 2.3.1 User account age at the time the order was placed
    df['user_account_age'] = calc_user_account_age(df)
    df_unknown['user_account_age'] = calc_user_account_age(df_unknown)

    # #### 2.3.2 User Age
    df['user_age'] = calc_user_age(df)
    df_unknown['user_age'] = calc_user_age(df_unknown)

    # #### 2.3.3 Number of orders by user at order time
    df['total_orders_by_user'] = calc_number_of_total_orders_by_user(df, user_known_dfs)
    df_unknown['total_orders_by_user'] = calc_number_of_total_orders_by_user(df_unknown, user_unknown_dfs)

    # #### 2.3.4 User has bought the same item before
    df['has_bought_item_before'] = calc_has_bought_item_before(df)
    df_unknown['has_bought_item_before'] = calc_has_bought_item_before(df_unknown)

    # #### 2.3.5 User's first purchase
    df['is_first_purchase'] = calc_is_first_purchase(df)
    df_unknown['is_first_purchase'] = calc_is_first_purchase(df_unknown)

    # #### 2.3.6 Number of items in order
    df['number_of_items_in_order'] = calc_number_of_items_in_order(df)
    df_unknown['number_of_items_in_order'] = calc_number_of_items_in_order(df_unknown)

    # #### 2.3.7 Ordered item at least twice in one order
    df['ordered_item_multiple_times_in_order'] = calc_ordered_item_multiple_times_in_order(df)
    df_unknown['ordered_item_multiple_times_in_order'] = calc_ordered_item_multiple_times_in_order(df_unknown)

    # ## 3 Feature Selection
    # ## 3.1 Unify Category Levels
    # ### 3.1.1 `item_id`
    df['item_id'], df_unknown['item_id'] = unify_cat_levels([df['item_id'], df_unknown['item_id']])

    # ### 3.1.1 `brand_id`
    df['brand_id'], df_unknown['brand_id'] = unify_cat_levels([df['brand_id'], df_unknown['brand_id']])

    # change dtype back to category
    df['brand_id'], df_unknown['brand_id']  = df['brand_id'].astype('category'), df_unknown['brand_id'].astype('category') 
    df['item_id'], df_unknown['item_id']  = df['item_id'].astype('category'), df_unknown['item_id'].astype('category') 
    
    # ## 3.2 Optimal Binning
    #columns for which we want to calculate woe_bins 
    woe_cols = ['item_id','brand_id','item_color','item_size']

    for col in woe_cols:
        optb = optimal_binning(df[col], df['return'])
        print(f'Status of Binning: {optb.status}')
        # use the inbuilt transform func of optb to automatically assign a binning group according to the woe value
        df[col+'_group'] = optb.transform(df[col], metric='indices')
        
        # transform unknown dataset cols
        df_unknown[col+'_group'] = optb.transform(df_unknown[col], metric='indices')
        
        #Change Dtypes
        df[col+'_group'] = df[col+'_group'].astype('category') 
        df_unknown[col+'_group'] = df_unknown[col+'_group'].astype('category') 
        
        #Save Bin Mapping in dict
        opt_b_df = optb.binning_table.build()
        bin_dict = {}
        for index, row in opt_b_df.iterrows():
            bin_dict[index] = list(row[0])
        
        #Save as JSON
        with open(f'../../data/03_processed/{col}_bin_map.json', 'w') as fp:
            json.dump(bin_dict, fp)



    # ## 4. Save Processed Data
    #Sanity #2: check to see if both df's have the same number of columns
    (len(df.columns)-1 == len(df_unknown.columns) ) & (len(df.columns)-1 == len(df_unknown.columns))

    # Save processed data with all feutures
    df.to_pickle('../../data/03_processed/BADS_WS2021_known_processed_all_cols.pkl')
    df_unknown.to_pickle('../../data/03_processed/BADS_WS2021_unknown_processed_all_cols.pkl')

