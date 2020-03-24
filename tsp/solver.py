#!/usr/bin/python
# -*- coding: utf-8 -*-

from anytree import Node, RenderTree
import math
import numpy as np
import random
from sklearn.metrics import pairwise_distances
import subprocess
import pandas as pd
from collections import namedtuple
from tqdm import tqdm


def unique_l(l):
    return list(set(l))

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
    #if verbose == 1:
    print(f"dist_to_alternatives before exclusions are: {', '.join([str(d) for d in dist_to_alternatives])}")
    print(f"exclusions: {exclude}")
    dist_to_alternatives[unique_l(exclude + [current_pt])] = np.inf
    neighbors_idx = np.argpartition(dist_to_alternatives, n)[:n]  # thanks https://stackoverflow.com/a/34226816/1870832
    #if verbose == 1:
    print(f"dist_to_alternatives after exclusions are: {', '.join([str(d) for d in dist_to_alternatives])}")
    print(f"neighbors_idx: {neighbors_idx}")
    return neighbors_idx


def heuristic_search_salesman(distance_matrix, startnode=0, n_closest=3, n_levels=3, verbose=0):
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
    print(f"Starting Heuristic Search Salesman for n_levels={n_levels} & n_closest={n_closest}")

    # input validation

    visit_order = [startnode]
    for i in tqdm(range(distance_matrix.shape[0]-1)):  # i is the tour position we're deciding now
        current_pt = visit_order[-1]

        # From current point, create a tree gaming out paths moving forward
        root = Node(str(current_pt))

        candidates = get_closest_nodes(current_pt, distance_matrix, n_closest, exclude=visit_order)
        nodes_by_tree_lvl = {k:[] for k in range(n_levels+1)}
        nodes_by_tree_lvl[0] = [Node(str(c), parent=root) for c in candidates]

        for level in range(1, n_levels):
            for candidate in nodes_by_tree_lvl[level-1]:
                if verbose == 1:
                    print('\n--------------------------------------------------------')
                    print(f"calculating for level {level} and candidate {candidate}")
                    print('--------------------------------------------------------')
                candidate_ancestors = [int(a.name) for a in candidate.ancestors]
                exclude = unique_l(visit_order + candidate_ancestors)
                if verbose == 1:
                    print(f"exclude-list for next candidates is: {exclude}")
                next_candidates = get_closest_nodes(int(candidate.name), distance_matrix, n_closest, exclude=exclude)
                if verbose == 1:
                    print(f"next candidates: {next_candidates}")
                    input("Press Enter to continue...")
                nodes_by_tree_lvl[level] = nodes_by_tree_lvl[level] + [Node(str(nc), parent=candidate) for nc in next_candidates]

        # Now that the heuristic search tree is constructed, calculate full distance for each path,
        # next step is first-step along shortest planned distance
        #print(RenderTree(root))
        next_step = np.nan
        shortest_dist = np.inf
        #distances = pd.DataFrame({'path':[], 'total_distance':[]})
        for possible_path in root.leaves:  # (Node('/0/1/3'), Node('/0/1/4'), Node('/0/2/5'), Node('/0/2/6'))
            nodes = [n.name for n in possible_path.ancestors] + [possible_path.name]
            dist = sum(distance_matrix[int(i),int(j)] for i,j in zip(nodes[0:-1],nodes[1:]))
            if verbose==1:
                #distances = pd.concat([distances, pd.DataFrame({'path'['/'.join(nodes), 'total_distance':[dist]})], axis=0)
                print(f"distance for {nodes} is {dist}")
            if dist < shortest_dist:
                shortest_dist = dist
                next_step = int(nodes[1])  # nodes[1] is second item in list, but first item is current-point

        print(f"next_step is: {next_step}")
        visit_order.append(next_step)

    return visit_order


def calc_tour_dist(tour_order, dist_matrix):

    # dist-matrix entry between each consecutive pair of stops in tour_order.
    # (plus last-entry back to first)
    total_dist = sum([dist_matrix[i,j] for i,j in zip(tour_order[:-1], tour_order[1:])])
    total_dist += dist_matrix[tour_order[-1], tour_order[0]]
    return total_dist


def solve_it(input_data, input_filename, n_levels=1, n_closest=3, n_starts=10, verbose=1):

    # Calculate distance matrix & initial route
    distance_matrix = get_dist_matrix(input_data)
    if verbose ==1:
        pd.DataFrame(distance_matrix, columns=[[str(i) for i in range(len(distance_matrix))]]).to_csv('distance_matrix.csv')

    # Starting towards a better initial solution using heuristic search trees
    tour = heuristic_search_salesman(distance_matrix, startnode=0, n_closest=n_closest, n_levels=n_levels, verbose=verbose)  # n_closest=1 would be original greedy (which is 506.36 on prob1)
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
        n_levels = int(sys.argv[2])
        n_closest = int(sys.argv[3])
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data, file_location, n_levels=n_levels, n_closest=n_closest))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

