#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import random
from sklearn.metrics import pairwise_distances
import subprocess
import pandas as pd
from collections import namedtuple
from tqdm import tqdm


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
    """ thanks to https://stackoverflow.com/a/60779058/1870832"""
    def __init__(self,nodeid):
        self.nodeid=nodeid
        self.child=[]

    def __str__(self):
        print(f"{self.nodeid}")

    def traverse(self, path = None):
        if path is None:
            path = []
        path.append(self.nodeid)
        if len(self.child) == 0:
            yield path
            path.pop()
        else:
            for child in self.child:
                yield from child.traverse(path)
            path.pop()


def heuristic_search_salesman(distance_matrix, startnode=0, n_closest=3, n_levels=2):
    '''At each node, consider a few closest next steps, then a few from there, etc.
        - Choose immediate next step which is on the way to best outcome gamed out n steps forward
        - See for ref Sec 8.9 "Reinforcement Learning," by Sutton and Barto

    params
    ------
    distance_matrix: self-explanatory
    startnode:       node to start the tour at 
    n_closest:       At each node, consider for next step this many (closest) nodes
    n_levels:        BASICALLY HARDCODED FOR NOW Plan this many levels/steps ahead before choosing next step
    drop_worst:      (NOT IMPLEMENTED) If True, do not consider in plan branches off worst node considered at each level 
    '''
    print(f"Starting Heuristic Search Salesman for n_closest={n_closest} and n_levels={n_levels}")

    # input validation
    assert n_levels in (2,3), "please keep n_levels as either 2 or 3"

    visit_order = [startnode]
    for i in tqdm(range(distance_matrix.shape[0]-1)):  # i is the tour position we're deciding now
        current_pt = visit_order[-1]

        # From current point, create a tree gaming out paths moving forward
        root = createnode(current_pt)

        ## first level of tree
        root.child += [createnode(node) for node in get_closest_nodes(current_pt, 
                                                                        distance_matrix,
                                                                        n_closest, 
                                                                        exclude=visit_order+[current_pt])]
        ## second level of tree
        for child in root.child:
            exclude_list = visit_order + [root.nodeid] + [child.nodeid]
            child.child += [createnode(node) for node in get_closest_nodes(child.nodeid, 
                                                                            distance_matrix,
                                                                            n_closest, 
                                                                            exclude=exclude_list)]

        # For all full root-leaf paths through the tree, select next step as first step along 
        # shortest planned path
        all_planned_paths = root.traverse()

        next_step = np.nan
        shortest_dist = np.inf
        for p in all_planned_paths:
            planned_dist = sum([distance_matrix[i,j] for i,j in zip(p[:-1], p[1:])])
            if planned_dist < shortest_dist:
                shortest_dist = planned_dist
                next_step = p[1]
        
        visit_order.append(next_step)

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

    # Starting towards a better initial solution using heuristic search trees
    tour = heuristic_search_salesman(distance_matrix, startnode=0, n_closest=3, n_levels=2)  # n_closest=1 would be original greedy (which is 506.36 on prob1)
    tour_dist = calc_tour_dist(tour, distance_matrix)


    """
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
    """

    # Format output as desired by course grader
    proved_opt=0
    output_data = f'{tour_dist:.2f} {proved_opt}\n'
    output_data += ' '.join(map(str, tour))
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

