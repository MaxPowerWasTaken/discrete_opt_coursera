#!/usr/bin/python
# -*- coding: utf-8 -*-
import io
import os
import subprocess
import pandas as pd
from  convert_data_to_dzn import convert_data_to_dzn

def solve_it(input_data, orig_filename='data.dzn'):
    #mz_model = "gc3.mzn -p 8" #"minizinc -p 8 graph_coloring_sym_breaking.mzn"
    #mz_options = " -l " #"-p 8 "

    # minizinc solver options by each problem (dict key is data line-1; nodes & edges)
    prob1 = {'solver': 'gecode', 'max-color': 6}   # each solver solves this quickly
    prob2 = {'solver': 'gecode', 'max-color': 17}  # solves in 25s, but ortools & chuffed hang for at least 2mins 
    prob3 = {'solver': 'ortools', 'max-color':16}  # solves in 33s, but gecode & chuffed hang at least 2mins
    prob4 = {'solver': 'chuffed', 'max-color':93}  # solves in 2.9s. gecode takes 173s (but solves) and ortools takes 62s
    prob5 = {'solver': 'gecode', 'max-color': 15}  # didn't write detailed notes on other solvers here, just not as good
    prob6 = {'solver': 'gecode', 'max-color': 122} # ditto above

    line1 = str.splitlines(input_data)[0]

    if "50 350" in line1: config_dict = prob1
    elif "70 1678" in line1: config_dict = prob2
    elif "100 2502" in line1: config_dict = prob3
    elif "250 28046" in line1: config_dict = prob4
    elif "500 12565" in line1: config_dict = prob5
    elif "1000 249482" in line1: config_dict = prob6
    else: 
        print("haven't seen this problem yet, not sure proper configuration")
        raise

    # Generate .dzn file for minizinc model
    print(orig_filename)
    convert_data_to_dzn(input_data, dzn_file = f"{orig_filename}.dzn")

    # Run Minizinc Model
    start = pd.Timestamp.utcnow()
    cmd_list = ["minizinc", "gc3.mzn", "-p", "8", "-D", f" 'MAX_COLOR={config_dict['max-color']};' ", 
                "-d", f"'{orig_filename}.dzn'", "--solver" ,f"org.{config_dict['solver']}.{config_dict['solver']}"]
    cmd_str = " ".join(cmd_list)

    sh_result = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE)
    print(f"finished minizinc run in {pd.Timestamp.utcnow()-start}")
    mz_output = sh_result.stdout.decode('UTF-8')
    print(mz_output)

    # Parse Minizinc Output and Prepare Coursera Course Output
    OPTIMAL = 0
    if "==========" in mz_output:
        OPTIMAL = 1

    colors = mz_output.split('\n')[0].strip('[]').split(', ')
    num_colors = len(set(colors))

    # prepare the solution in the specified output format
    output_data = f"{num_colors} {OPTIMAL}\n"
    output_data += ' '.join(colors)
    return output_data


import sys
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        # file_location = "data/gc_100_5"
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data, orig_filename=file_location))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

# cmd_list by problem:
#  for $prob2 ["minizinc", "gc3.mzn", "-p", "8", "-D", " 'MAX_COLOR=17;' ", "-d", f"{orig_filename}.dzn"]
#  for $prob3 ["minizinc", "gc3.mzn", "-p", "8", "-s", "-D", " 'MAX_COLOR=16;' ", "-d", "data/gc_100_5.dzn", "--solver" ,"org.ortools.ortools"]
# 