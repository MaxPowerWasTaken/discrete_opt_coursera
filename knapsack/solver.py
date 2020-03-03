#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm

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


def dp_select(values, weights, capacity):
    """Use dynamic programming to build up to solution for all items"""
    
    # convert params to n,K notation as used in equations in dynamic programming notes
    n = len(values)
    K = capacity
    
    # calculate table of optimal value by j,k (j items in 0..n, k is all capacities in 0..K)
    # see 9:00 - 12:30 here: https://www.coursera.org/learn/discrete-optimization/lecture/wFFdN/knapsack-4-dynamic-programming
    values_table = np.zeros((K+1,n+1), dtype=np.uint32)
    
    print("building DP optimal-value table for n,K: ", n, ",", K)
    for j in tqdm(range(1, n+1)):
        item_weight = weights[j-1]
        item_value  = values[j-1]
        for k in range(1, K+1):
            if item_weight > k:
                values_table[k,j] = values_table[k, j-1]
            else:
                values_table[k,j] = max(values_table[k, j-1], item_value + values_table[k-item_weight, j-1])
    optimal_value = values_table[-1, -1]
    print(f"optimal value is {optimal_value}. Now proceeding to derive final item-set")

    # from this table of optimal values, we now need to derive final item-set for optimal solution
    # logic of code below explained 12:30 - 14:00 at https://www.coursera.org/learn/discrete-optimization/lecture/wFFdN/knapsack-4-dynamic-programming
    taken = [0] * len(values)
    k = K   # in keeping w/ eqs, K is total capacity but k is k'th row as we move through j,k table
    for j in range(n, 0, -1):
        if values_table[k,j] != values_table[k,j-1]:
            taken[j-1] = 1
            k = k - weights[j-1]
    
    return optimal_value, taken


def solve_it(input_data):
    
    # parse the input
    (values, weights, K) = parse_data(input_data)
    
    # value-dense greedy algorithm for filling the knapsack
    obj, item_selections = dp_select(values, weights, K)

    # prepare the solution in the specified output format
    PROVED_OPTIMAL = 1
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

