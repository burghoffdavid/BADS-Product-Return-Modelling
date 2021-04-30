import pandas as pd
def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.5f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def fix_spelling_mistakes(ser, spelling_mistakes_dict):
    ser = ser.apply(lambda x: spelling_mistakes_dict[str(x)] if x in spelling_mistakes_dict.keys() else x)
    return ser.astype('category')


def make_lowercase(series_arr):
    for i in range(len(series_arr)):
        if isinstance(series_arr[i].values[0], str): 
            series_arr[i] = series_arr[i].str.lower().astype('category')
    return series_arr

