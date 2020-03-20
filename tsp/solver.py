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


def get_closest_nodes(current_pt, dist_matrix, n, exclude=[]):
    dist_to_alternatives = dist_matrix[current_pt].copy()
    dist_to_alternatives[exclude + [current_pt]] = np.inf
    neighbors_idx = np.argpartition(dist_to_alternatives, n)[:n]  # thanks https://stackoverflow.com/a/34226816/1870832
    return neighbors_idx


class createnode:
    """ thanks to https://stackoverflow.com/a/51911296/1870832"""
    def __init__(self,nodeid):
        self.nodeid=nodeid
        self.child=[]
    
    def __str__(self):
        print(f"{self.nodeid}")

    def traverse(self, path = []):
        paths=[[]]
        counter=0
        path.append(self.nodeid)
        if len(self.child) == 0:
            print(path)
            paths[counter] = path
            print(paths)
            counter+=1
            path.pop()
        else:
            for child in self.child:
                child.traverse(path)
            path.pop()

        return paths

# QUESTION WITH TREES:
    # do we need to calc dist at each node? or should we just
    # create structure, then take all paths, then calc distances
    # for all those paths
        # leaning second way

    # Second issue: arbitrary num levels and arbitrary num neighbors?
        # hm num-levels probably handled by caller that constructs tree
        # hum neighbors is num branches, so tree needs to handle arb number
            # but really in practice would only use 2 or 3?


def heuristic_search_salesman(distance_matrix, startnode=0, n_closest=2, n_levels=2):
    '''At each node, consider a few closest next steps, then a few from there, etc.
        - Choose immediate next step which is on the way to best outcome gamed out n steps forward
        - See for ref Sec 8.9 "Reinforcement Learning," by Sutton and Barto

    params
    ------
    distance_matrix: self-explanatory
    startnode:       node to start the tour at 
    n_closest:       At each node, consider for next step this many (closest) nodes
    n_levels:        Plan this many levels/steps ahead before choosing next step
    drop_worst:      (NOT IMPLEMENTED) If True, do not consider in plan branches off worst node considered at each level 
    '''
    visit_order = [startnode]
    for i in range(distance_matrix.shape[0]-1):  # i is the tour position we're deciding now
        current_pt = visit_order[-1]

        # for each path need to track: 1) nodes along path, 2) agg distance
        subpath=[current_pt]
        subpath_dist = 0
        subpath_results = {}




"""
           node=0
         /   |   \
        33   5    32     lvl = 0 (of 0,1,2)
      / | \ 
    26  22  5 - how to recurse...
  / | \
  5 6  7f
"""

def calc_tour_dist(tour_order, dist_matrix):

    # dist-matrix entry between each consecutive pair of stops in tour_order.
    # (plus last-entry back to first)
    total_dist = sum([dist_matrix[i,j] for i,j in zip(tour_order[:-1], tour_order[1:])])
    total_dist += dist_matrix[tour_order[-1], tour_order[0]]
    return total_dist


def solve_it(input_data, input_filename, n_starts=10):

    # Calculate distance matrix & initial route
    distance_matrix = get_dist_matrix(input_data)

    # Starting towards a better initial solution using heuristic search trees
    # (WIP - not done)
    ROOT_NODE = 0
    root = createnode(ROOT_NODE)
    children = [createnode(node) for node in get_closest_nodes(0, distance_matrix, n=2)]
    
    root.child += children 
    paths = root.traverse()  # prints full paths.
    print(paths)
    import sys; sys.exit(0)

    # Greedy Solution
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
        #file_location = "data/tsp_51_1"
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data, file_location))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

