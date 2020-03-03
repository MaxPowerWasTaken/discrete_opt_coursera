#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def unzip(iterable):
    ''' thanks to https://stackoverflow.com/a/22115957/1870832'''
    return zip(*iterable)


def parse_data(input_data):
    ''' Convert string input into set of np arrays, along with capacity'''

    lines = str.splitlines(input_data)

    # first line is n,K
    capacity = int(lines[0].split()[1])

    # convert list of 'value weight' strings to  unzipped lists of values, weights
    pairs = [x.split() for x in lines[1:]]  
    values, weights = unzip(pairs)         

    # convert from str to int
    values = [int(v) for v in values]
    weights = [int(w) for w in weights]

    return values, weights, capacity


def greedy_select(values, weights, K):
    ''' Value-dense greedy selection algorithm'''
    
    # convert to DF, add new column for value-density 
    df = pd.DataFrame({'value': values, 'weight': weights, 'capacity': K})
    df['value_density'] = df.value / df.weight

    # identify which items to select (top items by value-density, 
    # until cumulative-weight > capacity)
    df = df.sort_values(by="value_density", ascending=False)
    df['cumulative_weight'] = df.weight.cumsum()

    # return whether to select each item, **in original order given**, along with total value
    df['item_selected'] = np.where(df.cumulative_weight <= df.capacity, 1, 0)

    total_value = df.loc[df.item_selected==1, 'value'].sum()
    item_selections = list(df.sort_index().loc[:, 'item_selected'])

    return total_value, item_selections

def solve_it(input_data):
    
    # parse the input
    (values, weights, K) = parse_data(input_data)
    
    # value-dense greedy algorithm for filling the knapsack
    obj, item_selections = greedy_select(values, weights, K)

    # prepare the solution in the specified output format
    PROVED_OPTIMAL = 0
    output_data = f"{obj} {PROVED_OPTIMAL}\n"
    output_data += ' '.join(str(i) for i in item_selections)
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        #file_location = "data/ks_30_0"
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

