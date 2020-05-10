#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from anytree import Node, RenderTree
from itertools import combinations
import math
import numpy as np
import random
from sklearn.metrics import pairwise_distances
import subprocess
import sys
from numba import jit
import pandas as pd
from pprint import PrettyPrinter
from tqdm import tqdm


def solve_it(input_data):
    """ call julia solver """
    with open("data/for_jl_solver", "w") as f:
        f.writelines(input_data)

    # subprocess.run("julia tsp.jl", shell=True)
    cmd_str = "julia tsp.jl"
    sh_result = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE)
    sh_output = sh_result.stdout.decode("UTF-8")

    return sh_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str, help="path to data file. required")
    args = parser.parse_args()

    with open(args.datafile, "r") as input_data_file:
        input_data = input_data_file.read()

    print(solve_it(input_data))
