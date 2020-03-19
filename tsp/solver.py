#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import random
from sklearn.metrics import pairwise_distances
import subprocess
import pandas as pd
from collections import namedtuple


def get_dist_matrix(input_data):
    lines = input_data.split('\n')
    xypairs = [i.split() for i in lines[1:-1]]  # first line is num-points, last line is blank
    dist_matrix = pairwise_distances(xypairs, metric='euclidean')
    return dist_matrix


def greedy_salesman(distance_matrix, startnode=0):
    '''Generate a tour by drawing edges to closest remaining point, for each point in succession'''
    visit_order = [startnode]
    for i in range(distance_matrix.shape[0]-1):
        current_pt = visit_order[-1]
        dist_to_alternatives = distance_matrix[current_pt]
        dist_to_alternatives[visit_order] = np.inf  # can't select a point we've already visited
        next_pt = np.argmin(dist_to_alternatives)   # greedy - pt with min dist among possible choices
        visit_order.append(next_pt)
    assert len(visit_order) == len(distance_matrix)
    return visit_order


def calc_tour_dist(tour_order, dist_matrix):

    # dist-matrix entry between each consecutive pair of stops in tour_order.
    # (plus last-entry back to first)
    total_dist = sum([dist_matrix[i,j] for i,j in zip(tour_order[:-1], tour_order[1:])])
    total_dist += dist_matrix[tour_order[-1], tour_order[0]]
    return total_dist


def solve_it(input_data, input_filename, n_starts=10):
    
    # Calculate distance matrix & initial route
    distance_matrix = get_dist_matrix(input_data)

    N = int(input_data.split('\n')[0])
    best_tour = []
    lowest_dist = np.inf
    for n in random.sample(range(N), N): #n_starts):
        greedy_tour = greedy_salesman(distance_matrix.copy(), startnode=n)  # if pass this w/o .copy(), distance_matrix actually is changed up here.
        soln_dist = calc_tour_dist(greedy_tour, distance_matrix)
        print(f"With starting node {n}, greedy solution yielded total-distance of {soln_dist:,}")
        if soln_dist < lowest_dist:
            lowest_dist = soln_dist
            best_tour = greedy_tour
    
    # Format output as desired by course grader
    proved_opt=0
    output_data = f'{lowest_dist:.2f} {proved_opt}\n'
    output_data += ' '.join(map(str, best_tour))
    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # file_location = "data/tsp_51_1"
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data, file_location))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

