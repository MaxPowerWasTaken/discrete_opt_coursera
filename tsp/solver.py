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
from tqdm import tqdm


def unique_l(l):
    return list(set(l))


def get_dist_matrix(input_data):
    """input_data comes in as raw multiline text string"""
    lines = input_data.split('\n')
    xypairs = [i.split() for i in lines[1:-1]]  # first line is num-points, last line is blank
    dist_matrix = pairwise_distances(xypairs, metric='euclidean')
    return dist_matrix

def get_closest_nodes(current_pt, dist_matrix, n, exclude=[], verbose=False):
    dist_to_alternatives = dist_matrix[current_pt].copy()

    # don't consider as neighbors any points already visited
    dist_to_alternatives[unique_l(exclude + [current_pt])] = np.inf
    n_valid = min(n, np.isfinite(dist_to_alternatives).sum())

    neighbors_idx = np.argpartition(dist_to_alternatives, n_valid)[:n_valid]  # argpartition like an argmin to return n smallest
    return neighbors_idx


def calc_tour_dist(tour_order, dist_matrix):

    # dist-matrix entry between each consecutive pair of stops in tour_order.
    # (plus last-entry back to first)
    total_dist = sum([dist_matrix[i,j] for i,j in zip(tour_order[:-1], tour_order[1:])])
    total_dist += dist_matrix[tour_order[-1], tour_order[0]]
    return total_dist


def heuristic_search_salesman(distance_matrix, 
                              startnode=0, 
                              breadth=3, 
                              depth=3, 
                              verbose=False, 
                              debug=False):
    ''' Build out a heuristic search tree considering possible paths forward. 
        Take first step along shortest planned path.
        See for ref Sec 8.9 "Reinforcement Learning," by Sutton and Barto: 
        http://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf

        (Note: if depth or breadth is 1, this reduces to regular greedy search)

    params
    ------
    distance_matrix: square matrix of distance from each point to each other point
    startnode:       node the TSP starts from 
    breadth:         breadth of the search tree (how many next-steps considered from each step)
    depth:           depth of the search tree (how many steps forward to plan)
    '''
    print(f"Starting Heuristic Search Salesman for depth={depth} & breadth={breadth}")

    visit_order = [startnode]
    for i in tqdm(range(distance_matrix.shape[0]-1)):  # i is the tour position we're deciding now
        current_pt = visit_order[-1]

        # From current point, create a tree gaming out paths moving forward
        root = Node(str(current_pt))
        
        # first level of planning tree: candidates for next-move from current point
        candidates = get_closest_nodes(current_pt, distance_matrix, breadth, exclude=visit_order)
        nodes_by_tree_lvl = {k:[] for k in range(depth+1)}
        nodes_by_tree_lvl[0] = [Node(str(c), parent=root) for c in candidates]

        # fill out rest of planning tree in a loop
        for level in range(1, depth):
            for candidate in nodes_by_tree_lvl[level-1]:
                candidate_ancestors = [int(a.name) for a in candidate.ancestors]
                exclude = unique_l(visit_order + candidate_ancestors)
                next_candidates = get_closest_nodes(int(candidate.name), distance_matrix, breadth, exclude=exclude)
                nodes_by_tree_lvl[level] = nodes_by_tree_lvl[level] + [Node(str(nc), parent=candidate) for nc in next_candidates]

        # Now that the heuristic search tree is constructed, calculate full distance for each potential path,
        # next-step will be first-step along shortest planned path
        next_step = np.nan
        shortest_dist = np.inf
        for possible_path in root.leaves:
            nodes = [n.name for n in possible_path.ancestors] + [possible_path.name]
            dist = sum(distance_matrix[int(i),int(j)] for i,j in zip(nodes[0:-1],nodes[1:]))

            # if nodes already visited + depth of planning tree extends to all nodes, need
            # to include distance back to start to complete circuit in path's planned dist
            if len(visit_order) + len(nodes)-1 == distance_matrix.shape[0]:
                distance_back_to_start = distance_matrix[startnode, int(nodes[-1])]
                dist = dist + distance_back_to_start
                
            if verbose:
                print(f"distance for {nodes} is {dist}")
            if dist < shortest_dist:
                shortest_dist = dist
                next_step = int(nodes[1])  # nodes[0] is current-point. so nodes[1] is next step

        visit_order.append(next_step)
        if verbose:
            print(f"{visit_order}, cumulative distance: {sum([distance_matrix[i,j] for i,j in zip(visit_order[:-1], visit_order[1:])])}")
        if debug:
            input("Press Enter to continue...")

    return visit_order


def solve_it(input_data, 
             depth=3, 
             breadth=3, 
             verbose=False, 
             debug=False):
    """ Run python solver.py -h from shell for explanations of parameters """

    # Calculate distance matrix. Optionally save to csv disk for debugging
    distance_matrix = get_dist_matrix(input_data)
    if verbose ==1:
        print("Saving Distance-Matrix for distances among all nodes to each other to distance_matrix.csv\n")
        pd.DataFrame(distance_matrix, columns=[[str(i) for i in range(len(distance_matrix))]]).to_csv('distance_matrix.csv')

    # Conduct heuristic search. Breadth or Depth of 1 reduces to regular greedy search
    start = 0
    tour = heuristic_search_salesman(distance_matrix, 
                                     startnode=start, 
                                     breadth=breadth,
                                     depth=depth, 
                                     verbose=verbose,
                                     debug=debug)  
    tour_dist = calc_tour_dist(tour, distance_matrix)

    # Format output as desired by course grader
    proved_opt=0
    output_data = f'{tour_dist:.2f} {proved_opt}\n'
    output_data += ' '.join(map(str, tour))
    return output_data


if __name__ == '__main__':
    # CLI Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str, help = "path to data file. required")
    parser.add_argument('-d', '--depth', type=int, default='3', 
                        help='Number of Levels to plan forward in heuristic search tree. 1 means regular greedy search')
    parser.add_argument('-b', '--breadth', type=int, default='3', 
                        help='Number of closest nodes to consider at each level of the heuristic search tree')
    parser.add_argument('-v', '--verbose', action="store_true", help="Show extra print statements")
    parser.add_argument('--debug', action="store_true", 
                        help="Pause execution until keypress after each next-step selection. Sets verbose to True as well")

    # Parse CLI args and call solver 
    args = parser.parse_args()

    with open(args.datafile, 'r') as input_data_file:
        input_data = input_data_file.read()

    print(solve_it(input_data,  
                  depth=args.depth, 
                  breadth=args.breadth,
                  verbose=max(args.verbose,args.debug),  # no point calling debug w/o verbose
                  debug=args.debug))