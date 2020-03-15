#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import namedtuple
from sklearn.metrics import pairwise_distances

#Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def get_dist_matrix(input_data):
    lines = input_data.split('\n')
    xypairs = [i.split() for i in lines[1:-1]]  # first line is num-points, last line is blank
    dist_matrix = pairwise_distances(xypairs, metric='euclidean')
    return dist_matrix


def greedy_salesman(distance_matrix):
    '''Generate a tour by drawing edges to closest remaining point, for each point in succession'''
    visited = [0]
    for i in range(distance_matrix.shape[0]-1):
        current_pt = visited[-1]
        dist_to_alternatives = distance_matrix[current_pt]
        dist_to_alternatives[visited] = np.inf  # can't select a point we've already visited
        next_pt = np.argmin(dist_to_alternatives)
        visited.append(next_pt)
    assert len(visited) == len(distance_matrix)
    return visited

def solve_it(input_data):

    # Calculate distance matrix & initial route
    distance_matrix = get_dist_matrix(input_data)
    s0 = greedy_salesman(distance_matrix.copy())  # if pass this w/o .copy(), distance_matrix actually is changed up here.

    # Improve tour
    final_tour = s0

    # Calculate total-distance traveled of final tour
    total_dist = sum([distance_matrix[i,j] for i,j in zip(final_tour[:-1], final_tour[1:])])
    total_dist += distance_matrix[final_tour[-1], final_tour[0]]

    # Format output as desired by course grader
    proved_opt=0
    output_data = f'{total_dist:.2f} {proved_opt}\n'
    output_data += ' '.join(map(str, final_tour))
    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

