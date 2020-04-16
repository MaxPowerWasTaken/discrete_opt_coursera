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
import pandas as pd
from pprint import PrettyPrinter
from tqdm import tqdm


def argmax_k(a, k, return_sorted=True):
    """ return k indices of k-largest elements in array a 
    
    Runs in O(n + k*log(k)) time if return_sorted = True, and O(n) time otherwise (n = len(a))
    according to https://stackoverflow.com/a/23734295/1870832
    """
    a = np.asarray(a)
    idx_top_k = np.argpartition(a, -k)[-k:]  # not sorted at this point
    if not return_sorted:
        return idx_top_k
    else:
        idx_topk_sorted_desc = idx_top_k[np.argsort(a[idx_top_k])]
        return idx_topk_sorted_desc[::-1]


assert np.array_equal(
    argmax_k(a=[5, 7, 4, 2, 8], k=2, return_sorted=True), np.array([4, 1])
)
assert set(argmax_k(a=[5, 7, 4, 2, 8], k=2, return_sorted=False)) == set([1, 4])

agk = argmax_k(a=np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0]), k=4, return_sorted=True)
assert (agk[0:3] == [0, 5, 8]).all() & (
    agk[3] in (1, 2, 8)
)  # index of 3 largest is 0,5,8. 3-way tie for 4th-largest, any of them will do


def matrix_argmax(m):
    """thanks to https://stackoverflow.com/a/53470929/1870832 """
    return np.unravel_index(np.argmax(m, axis=None), m.shape)  # returns a tuple


def unique_l(l):
    return list(set(l))


def get_dist_matrix(input_data):
    """input_data comes in as raw multiline text string"""
    lines = input_data.split("\n")
    xypairs = [
        i.split() for i in lines[1:-1]
    ]  # first line is num-points, last line is blank
    dist_matrix = pairwise_distances(xypairs, metric="euclidean")
    return dist_matrix


def get_closest_nodes(current_pt, dist_matrix, n, exclude=[], verbose=False):
    dist_to_alternatives = dist_matrix[current_pt].copy()

    # don't consider as neighbors any points already visited
    dist_to_alternatives[unique_l(exclude + [current_pt])] = np.inf
    n_valid = min(n, np.isfinite(dist_to_alternatives).sum())

    neighbors_idx = np.argpartition(dist_to_alternatives, n_valid)[
        :n_valid
    ]  # argpartition like an argmin to return n smallest
    return neighbors_idx


def calc_tour_dist(tour_order, dist_matrix):

    # dist-matrix entry between each consecutive pair of stops in tour_order.
    # (plus last-entry back to first)
    total_dist = sum(
        [dist_matrix[i, j] for i, j in zip(tour_order[:-1], tour_order[1:])]
    )
    total_dist += dist_matrix[tour_order[-1], tour_order[0]]
    return total_dist


def greedy_salesman(distance_matrix, startnode=0):
    """Generate a tour by drawing edges to closest remaining point, for each point in succession"""
    dm = distance_matrix.copy()
    visit_order = [startnode]
    for i in range(dm.shape[0] - 1):
        current_pt = visit_order[-1]
        dist_to_alternatives = dm[current_pt]
        # dont wanna select a point we've already visited
        dist_to_alternatives[visit_order] = np.inf
        # greedy - pt with min dist among possible choices
        next_pt = np.argmin(dist_to_alternatives)
        visit_order.append(next_pt)
    assert len(visit_order) == len(distance_matrix)
    return visit_order


def dist_around(tour, nodeidx, dist_matrix):

    # next node for last node is first elem (idx=0)
    prev_idx = (nodeidx - 1) % len(tour)
    next_idx = (nodeidx + 1) % len(tour)

    d1 = dist_matrix[tour[prev_idx], tour[nodeidx]]
    d2 = dist_matrix[tour[nodeidx], tour[next_idx]]
    return d1 + d2


def two_opt(tour, dist_matrix, verbose=False):

    for i, swap_option in tqdm(enumerate(combinations(range(len(tour)), 2))):
        idx1, idx2 = swap_option

        tour2 = tour.copy()
        tour2[idx1], tour2[idx2] = tour2[idx2], tour2[idx1]

        old_dist = dist_around(tour, idx1, dist_matrix) + dist_around(
            tour, idx2, dist_matrix
        )

        new_dist = dist_around(tour2, idx1, dist_matrix) + dist_around(
            tour2, idx2, dist_matrix
        )

        dist_improvement = old_dist - new_dist

        if dist_improvement > 0:
            if verbose:
                msg = f"After {i} swap_options, swapped positions {idx1} & {idx2} for improvement of {dist_improvement}"
                print(msg)
            return (tour2, dist_improvement)

    if verbose:
        print(f"no 2opt combinations improved, hit local optimum")
    return None


"""
idx1
0
idx2
1
dist_around(tour2, idx1, dist_matrix)
inf"""


def solve_it(input_data, num_starts=1, verbose=False, debug=False):
    """ Run python solver.py -h from shell for explanations of parameters """

    # Calculate distance matrix. Optionally save to csv disk for debugging
    distance_matrix = get_dist_matrix(input_data)

    # get starting tour using regular greedy
    best_tour = None
    best_dist = np.inf

    if num_starts == 1:
        starts = [0]
    else:
        starts = np.random.choice(
            range(len(distance_matrix)), num_starts, replace=False
        )
    for start in tqdm(starts):
        tour = greedy_salesman(distance_matrix, startnode=start)
        tour_dist = calc_tour_dist(tour, distance_matrix)

        # Improve solution with local moves
        NUM_MOVES = 1000
        for n in range(NUM_MOVES):

            resp = two_opt(tour, distance_matrix, verbose)
            if resp is not None:
                tour, dist_improved = resp
            else:
                break

        tour_dist = calc_tour_dist(tour, distance_matrix)
        if verbose:
            print(f"final tour dist for startnode {start}: {tour_dist}")

        if debug:
            input("Press Enter to continue...")

        if tour_dist < best_dist:
            best_dist = tour_dist
            best_tour = tour

    # Format output as desired by course grader
    proved_opt = 0
    output_data = f"{best_dist:.2f} {proved_opt}\n"
    output_data += " ".join(map(str, best_tour))
    return output_data


if __name__ == "__main__":
    # CLI Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str, help="path to data file. required")
    parser.add_argument(
        "-n",
        "--numstarts",
        type=int,
        default=1,
        help="Number of different nodes to restart initial tour from",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show extra print statements"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pause execution until keypress after each next-step selection. Sets verbose to True as well",
    )

    # Parse CLI args and call solver
    # from box import Box; args = Box()
    args = parser.parse_args()

    with open(args.datafile, "r") as input_data_file:
        input_data = input_data_file.read()

    print(
        solve_it(
            input_data,
            verbose=max(
                args.verbose, args.debug
            ),  # no point calling debug w/o verbose
            debug=args.debug,
        )
    )
