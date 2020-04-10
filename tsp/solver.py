#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from anytree import Node, RenderTree
import math
import numpy as np
import random
from sklearn.metrics import pairwise_distances
import subprocess
import sys
import pandas as pd
from pprint import PrettyPrinter
from collections import namedtuple
from tqdm import tqdm


def unique_l(l):
    return list(set(l))


def get_dist_matrix(input_data):
    lines = input_data.split('\n')
    xypairs = [i.split() for i in lines[1:-1]]  # first line is num-points, last line is blank
    dist_matrix = pairwise_distances(xypairs, metric='euclidean')
    return dist_matrix


def get_closest_nodes(current_pt, dist_matrix, n, exclude=[], verbose=False):
    dist_to_alternatives = dist_matrix[current_pt].copy()
    dist_to_alternatives[unique_l(exclude + [current_pt])] = np.inf

    n_valid = min(n, np.isfinite(dist_to_alternatives).sum())  # dont return any .infs from exclude-list
    neighbors_idx = np.argpartition(dist_to_alternatives, n_valid)[:n_valid]  # thanks https://stackoverflow.com/a/34226816/1870832
    if verbose:
        print(f"exclusions: {exclude}")
        print(f"dist_to_alternatives are: {', '.join([str(d) for d in dist_to_alternatives])}")
        print(f"neighbors_idx: {neighbors_idx}")
    return neighbors_idx


def heuristic_search_salesman(distance_matrix, 
                              startnode=0, 
                              n_closest=3, 
                              n_levels=3, 
                              verbose=False, 
                              debug=False):
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

    visit_order = [startnode]
    for i in tqdm(range(distance_matrix.shape[0]-1)):  # i is the tour position we're deciding now
        current_pt = visit_order[-1]

        # From current point, create a tree gaming out paths moving forward
        root = Node(str(current_pt))
        
        # first level of tree for candidates for next-move from current point
        candidates = get_closest_nodes(current_pt, distance_matrix, n_closest, exclude=visit_order)
        nodes_by_tree_lvl = {k:[] for k in range(n_levels+1)}
        nodes_by_tree_lvl[0] = [Node(str(c), parent=root) for c in candidates]  # THIS ADDS TO TREE

        # loop, for each level, consider candidate nodes from each candidate node from previous level in search tree
        pp = PrettyPrinter(indent=4)
        for level in range(1, n_levels):
            for candidate in nodes_by_tree_lvl[level-1]:
                #if verbose:
                    #pp.pprint(nodes_by_tree_lvl)
                    #print('\n--------------------------------------------------------')
                #    print(f"calculating for level {level} and candidate {candidate}")
                    #print('--------------------------------------------------------')
                    #print(f"visited_already is: {visit_order}\n")
                candidate_ancestors = [int(a.name) for a in candidate.ancestors]
                exclude = unique_l(visit_order + candidate_ancestors)
                #if verbose:
                #    print(f"exclude-list for next candidates is: {exclude}")

                next_candidates = get_closest_nodes(int(candidate.name), distance_matrix, n_closest, exclude=exclude)
                if verbose:
                    print(f"next candidates: {next_candidates}")
                    #print(RenderTree(root))
                #if debug:
                    #input("Press Enter to continue...")
                # Add new candidate nodes to next level of heuristic search tree
                nodes_by_tree_lvl[level] = nodes_by_tree_lvl[level] + [Node(str(nc), parent=candidate) for nc in next_candidates]
                #print(RenderTree(root))
        # Now that the heuristic search tree is constructed, calculate full distance for each path,
        # next-step will be first-step along shortest planned path in search tree
        next_step = np.nan
        shortest_dist = np.inf
        for possible_path in root.leaves:  #root.leaves looks like: (Node('/0/1/3'), Node('/0/1/4'), Node('/0/2/5'), Node('/0/2/6'))
            nodes = [n.name for n in possible_path.ancestors] + [possible_path.name]
            dist = sum(distance_matrix[int(i),int(j)] for i,j in zip(nodes[0:-1],nodes[1:]))
            #print(nodes)
            #print(f"len of visit-order + nodes is {len(visit_order) + len(nodes)-1}, compared to {distance_matrix.shape[0]}")

            # nodes includes prospective candidate paths, but also current node which is last item in visit order
            if len(visit_order) + len(nodes)-1 == distance_matrix.shape[0]:
                print(f"adding {distance_matrix[int(nodes[-1]), startnode]} as dist from {nodes[-1]} back to {startnode} to path {nodes}")
                distance_back_to_start = distance_matrix[startnode, int(nodes[-1])]
                dist = dist + distance_back_to_start
                if debug:
                    input("Press Enter to continue...")
            
            # HERE IS WHERE I SHOULD CONDITIONALLY UPDATE DIST TO INCLUDE LAST BACK TO FIRST (IF PATH IS LEN OF ALL POINTS)


            #if verbose:
            #    print(f"distance for {nodes} is {dist}")
            if dist < shortest_dist:
                shortest_dist = dist
                next_step = int(nodes[1])  # nodes[1] is second item in list, but first item is current-point. so nodes[1] is next step

        #print(f"next_step is: {next_step}")
        visit_order.append(next_step)

    return visit_order


def calc_tour_dist(tour_order, dist_matrix):

    # dist-matrix entry between each consecutive pair of stops in tour_order.
    # (plus last-entry back to first)
    total_dist = sum([dist_matrix[i,j] for i,j in zip(tour_order[:-1], tour_order[1:])])
    total_dist += dist_matrix[tour_order[-1], tour_order[0]]
    return total_dist







def solve_it(input_data, 
             input_filename, 
             n_levels=3, 
             n_closest=3, 
             verbose=False, 
             debug=False):
    """ Run python solver.py -h from shell for explanations of parameters """

    # Calculate distance matrix. Optionally save to csv disk for debugging
    distance_matrix = get_dist_matrix(input_data)
    if verbose ==1:
        print("Saving Distance-Matrix for distances among all nodes to each other to distance_matrix.csv\n")
        pd.DataFrame(distance_matrix, columns=[[str(i) for i in range(len(distance_matrix))]]).to_csv('distance_matrix.csv')

    # Conduct heuristic search
    start = 0
    tour = heuristic_search_salesman(distance_matrix, 
                                     startnode=start, 
                                     n_closest=n_closest, # n_closest=1 would be original greedy (which is 506.36 on data/tsp_51_1) 
                                     n_levels=n_levels, 
                                     verbose=verbose,
                                     debug=debug)  
    tour_dist = calc_tour_dist(tour, distance_matrix)


    """ Code below for trying heuristic search with restarts.
        Setting this aside for now while debugging heuristic search from single start

    best_tour = []
    lowest_dist = np.inf
    for start in (0,2,5,17,21,35,44):
        tour = heuristic_search_salesman(distance_matrix, startnode=start, n_closest=n_closest, n_levels=n_levels, verbose=verbose)  # n_closest=1 would be original greedy (which is 506.36 on prob1)
        tour_dist = calc_tour_dist(tour, distance_matrix)
        print(f"for start at {start}, tour distance = {tour_dist}")

        if tour_dist < lowest_dist:
            best_tour = tour
            lowest_dist = tour_dist
    """

    # Format output as desired by course grader
    proved_opt=0
    output_data = f'{tour_dist:.2f} {proved_opt}\n'
    output_data += ' '.join(map(str, tour))
    return output_data


import sys

if __name__ == '__main__':
    # CLI Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str, help = "path to data file. required")
    parser.add_argument('-l', '--levels', type=int, default='3', 
                        help='Number of Levels to plan forward in heuristic search tree. 1 means regular greedy search')
    parser.add_argument('-c', '--nclose', type=int, default='3', 
                        help='Number of closest nodes to consider at each level of the heuristic search tree')
    parser.add_argument('-v', '--verbose', action="store_true", help="Show extra print statements")
    parser.add_argument('-d', '--debug', action="store_true", 
                        help="Pause execution until keypress after each next-step selection. Sets verbose to True as well")

    # Parse CLI args and call solver 
    args = parser.parse_args()
    file_location = args.datafile
    n_levels = args.levels
    n_closest = args.nclose
    verbose = args.verbose
    debug = args.debug
    if debug:
        verbose=True  #

    # file_location = 'data/tsp_51_1'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()

    print(solve_it(input_data, 
                  file_location, 
                  n_levels=n_levels, 
                  n_closest=n_closest,
                  verbose=verbose,
                  debug=debug))