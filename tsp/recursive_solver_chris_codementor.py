#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from math import sqrt
from bisect import insort

def get_dist_matrix(input_data):
    lines = input_data.split('\n')[1:-1]  # first line is num-points, last line is blank
    xy = [tuple(map(float, line.split())) for line in lines]
    n = len(xy)

    def dist(i, j):
        dx = xy[i][0] - xy[j][0]
        dy = xy[i][1] - xy[j][1]
        return sqrt(dx*dx + dy*dy)

    return [[dist(i,j) for j in range(n)] for i in range(n)]

def closest(n, matrix, path):
    """ mwe: returns closest node(c) to current node(n), given dist-matrix and path visited-so-far """
    c = -1
    m = float('inf')
    for o in range(len(matrix)):
        if o not in path:
            d = matrix[n][o]
            if d < m:
                m = d
                c = o
    return c

class PathFinder:

    def __init__(self, depth, breadth):
        self.depth = depth
        self.breadth = breadth

    def closest(self, n, matrix, path):
        c = []
        m = float('inf')
        for o in range(len(matrix)):
            if o not in path:
                d = matrix[n][o]
                if d < m:
                    insort(c, (d, o))
                    if len(c) > self.breadth:
                        c.pop()
                        m = c[-1][0]
        return c

    def find_rec(self, l, n, matrix, path, dist):
        if dist > self.best_dist: return
        if l < self.depth and len(path) < len(matrix):
            for _, o in self.closest(n, matrix, path):
                path.append(o)
                if l == 0: self.start = o
                self.find_rec(l + 1, o, matrix, path, dist + matrix[n][o])
                path.pop()
        else:
            if dist < self.best_dist:
                self.best_dist = dist
                self.best_start = self.start

    def find(self, n, matrix, path):
        self.best_dist = float('inf')
        self.find_rec(0, n, matrix, path, 0)
        return self.best_start

class RouteFinder:

    def __init__(self, best_fn):
        self.best_fn = best_fn

    def find_rec(self, n, matrix, path, dist):
        path.append(n)
        #print(path)
        if len(path) < len(matrix):
            o = self.best_fn(n, matrix, path)
            self.find_rec(o, matrix, path, dist + matrix[n][o])
        else:
            dist += matrix[n][self.start]
            if dist < self.best_dist:
                self.best_dist = dist
                self.best_path = path.copy()
        path.pop()

    def find(self, matrix):
        """ mwe: this just calls find_rec() for each different node to find best starting position"""
        self.best_dist = float('inf')
        for n in range(len(matrix)):
            self.start = n
            self.find_rec(n, matrix, [], 0)

def solve_it(matrix, best_fn):
    tsp = RouteFinder(best_fn)
    tsp.find(matrix)
    print(tsp.best_dist)
    print(tsp.best_path)

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

    with open(args.datafile, 'r') as input_data_file:
        input_data = input_data_file.read()
    matrix = get_dist_matrix(input_data)

    solve_it(matrix, closest)  # mwe: closest is a func. regular greedy search
                               # so solve_it calls RouteFinder w/ greedy-search as best-neighbor function

    solve_it(matrix, PathFinder(args.levels, args.nclose).find)
                               # so solve_it calls RouteFinder with best-neighbor function of Pathfinder.find