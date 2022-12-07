# this is the script for getting dimensions of the grids

import random

def find_max_min_row(df):
    find_max_row = float('-inf')
    find_min_row = float('inf')
    for index, a_row in df.iterrows():
        #print(a_row.free_part)
        for a_key in eval(a_row.free_part):
           #print(a_key[0])
            if a_key[0] < find_min_row:
                find_min_row = a_key[0]
            if a_key[0] > find_max_row:
                find_max_row = a_key[0]
    return find_max_row, find_min_row

def find_max_min_col(df):
    find_max_column = float('-inf')
    find_min_column = float('inf')
    for index, a_row in df.iterrows():
        #print(a_row.free_part)
        for a_key in eval(a_row.free_part):
           #print(a_key[0])
            if a_key[1] < find_min_column:
                find_min_column = a_key[1]
            if a_key[1] > find_max_column:
                find_max_column = a_key[1]
    return find_max_column, find_min_column

def find_max_min_jones(df):
    find_max_column = float('-inf')
    find_min_column = float('inf')
    for index, a_row in df.iterrows():
        #print(a_row.free_part)
        for a_key in eval(a_row.polynomial).keys():
           #print(a_key[0])
            if a_key < find_min_column:
                find_min_column = a_key
            if a_key > find_max_column:
                find_max_column = a_key
    return find_max_column, find_min_column
