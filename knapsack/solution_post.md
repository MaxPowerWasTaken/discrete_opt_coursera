This is the first in (hopefully) a series of blog posts on solutions to weekly programming assignments for [Professor Pascal Van Hentenryck's Discrete Optimization course on Coursera](https://www.coursera.org/learn/discrete-optimization/home/welcome). In my experience so far, the lectures are very good, and the programming assignments are even better. 

Each week, Professor Van Hentenryck introduces a set of algorithmic approaches to combinatorial optimization problems. Then, the weekly programming assignment is an NP-hard optimization problem, along with a set of several different input data files. The different data files are all valid inputs to the same problem, but of different sizes (i.e. it's much harder to find an optimal solution for a given NP-hard problem on a large dataset than a small one)

It's always going to be pretty easy to get a pretty good solution for the smaller data files. To get an optimal solution to the smaller data files, you might have to be a bit more clever. To get a good solution to the larger data files, it's gonna be a bit more work. And to do really well, you need to find optimal (or at least close to optimal) solutions for even the largest data files. 

## The First Week's Programming Assignment

The first week's programming assingment is the [Knapsack Problem](https://en.wikipedia.org/wiki/Knapsack_problem). 

### Problem Formulation
Formally, the problem is defined as:

maximize:  $$\sum_{i \in 0...n-1} v_i x_i$$

subject to: $\sum_{i in 0...n-1} w_i x_i \leq K   $, $x \in {0,1}$

If this looks confusing, here's the plain english interpretation of the problem:
- We are presented with a set of items, and a knapsack
- Each item has a value $(v)$ and weight $(w)$. Our knapsack can hold a maximum combined weight of $K$
- $x_i$ is 0 or 1 depending on if we place item $i$ in our knapsack

Hopefully now the math above looks pretty straightforward: find the set of items that maximizes their combined value, such that the combined weights of the selected items don't exceed our knapsack weight capacity.

### Data Format Specification
Each input file we're given has n+1 lines in the following format:

```
n K 
v_0 w_0
v_1 w_1
...
v_n-1 w_n-1
```

...where the $n$ and $K$ on the top line represent the number of items we have to select from, and the capacity of the knapsack. Each line after the first describes an item; $v_i$ & $ w_i$

For our solution, we'll output two lines of the format:

```
obj opt 
x_0 x_1 ... x_n-1
```
...where:
- 'obj' will be the total value of our selected items, 
- 'opt' is 1 if we know our solution is optimal, and 0 otherwise
- x_i... are all 0 or 1 for whether we selected that item or not  

## Jumping into the code
As a jumping off point, [we're given the following naive solution](https://github.com/MaxPowerWasTaken/discrete_opt_coursera/blob/6d8f8a854b1a6a13a5b22d6bf9997ed2d63ca49d/knapsack/solver.py#L7):

```python
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data
```


...there's a bit more scaffolding code as well, namely a `submit.py` script that calls this `solve_it` function on progressively larger input data files, sends the solutions to coursera's server and then records the scores.
``` ```   

...As a quick aside, I much prefer the user experience for the programming assignments in this course-- programming in my own editor/environment, and submitting my solutions via a script from my own terminal-- to the alternative that some MOOCs have, where all programming assignments are done in a jupyter notebook on their server.

Anyway, the naive solution starter code we were given scores 3/10 on each of the six problems, for 18/60 overall.

### Coding our first solution 

The optimization topics covered in this first week were:
- Greedy algorithms
- Dynamic Programming
- Relaxation, Branch and Bound

So for my first solution, as a baseline, I started with a simple value-dense greedy selection:


```python
import numpy as np
import pandas as pd

def unzip(iterable):
    ''' thanks to https://stackoverflow.com/a/22115957/1870832'''
    return zip(*iterable)


def parse_data(input_data):
    ''' Convert string input into set of np arrays, along with capacity'''

    lines = str.splitlines(input_data)

    # first line is n,K
    capacity = int(lines[0].split()[1])

    # convert list of 'value weight' strings to  unzipped lists of values, weights
    pairs = [x.split() for x in lines[1:]]  
    values, weights = unzip(pairs)         

    # convert from str to int
    values = [int(v) for v in values]
    weights = [int(w) for w in weights]

    return values, weights, capacity


def greedy_select(values, weights, K):
    ''' Value-dense greedy selection algorithm'''
    
    # convert to DF, add new column for value-density 
    df = pd.DataFrame({'value': values, 'weight': weights, 'capacity': K})
    df['value_density'] = df.value / df.weight

    # identify which items to select (top items by value-density, 
    # until cumulative-weight > capacity)
    df = df.sort_values(by="value_density", ascending=False)
    df['cumulative_weight'] = df.weight.cumsum()

    # return whether to select each item, **in original order given**, along with total value
    df['item_selected'] = np.where(df.cumulative_weight <= df.capacity, 1, 0)

    total_value = df.loc[df.item_selected==1, 'value'].sum()
    item_selections = list(df.sort_index().loc[:, 'item_selected'])

    return total_value, item_selections

def solve_it(input_data):
    
    # parse the input
    (values, weights, K) = parse_data(input_data)
    
    # value-dense greedy algorithm for filling the knapsack
    obj, item_selections = greedy_select(values, weights, K)

    # prepare the solution in the specified output format
    PROVED_OPTIMAL = 0
    output_data = f"{obj} {PROVED_OPTIMAL}\n"
    output_data += ' '.join(str(i) for i in item_selections)
    return output_data

```

If you want to see the diff of the original `solve_it` file to the value-dense greedy version above, [you can check it out here](https://github.com/MaxPowerWasTaken/discrete_opt_coursera/commit/9e9f4e87df79fd704b9ec6ee045e7ff0d7ce0f4c#diff-57214dafea7890eb954c73d3f58b3d17)

Alright so how does this solution do? We can check by running the `submit.py` script we're given as part of the assignment, which will run our `solve_it` code for six incrementally larger data files, send the solutions to the coursera server, where it will be scored:

```bash
(knapenv) maxepstein@pop-os:~/discrete_opt/knapsack$ python submit.py
==
== Knapsack Solution Submission 
==
Hello! These are the assignment parts that you can submit:
1) Knapsack Problem 1
2) Knapsack Problem 2
3) Knapsack Problem 3
4) Knapsack Problem 4
5) Knapsack Problem 5
6) Knapsack Problem 6
0) All
Please enter which part(s) you want to submit (0-6): 0
```
...As a quick aside, I much prefer the user experience for the programming assignments in this course-- programming in my own editor/environment, and submitting my solutions via a script from my own terminal-- to the alternative that some MOOCs have, where all programming assignments are done in a jupyter notebook on their server.

Anyway, back to the scoring. My value-dense greedy solution scores 3/10 on each of the six problems, for 18/60 overall. That's lower than I expected, which is actually encouraging; we'll have to do something reasonable to get a non-terrible score. Let's move on to a non-baseline solution.

## Using Dynamic Programming.
Dynamic Programming(DP) has the advantage of guaranteeing an optimal solution...if we have the compute time & RAM to complete the solution. 

DP is kinda like recursion, in that we use a set of `recurrence relations` (Bellman Equations) to define how we can break a given problem into smaller sub-problems. We also have a solution for the simplest, sorta atomic level of subproblem. However, recursion is top-down, in that you start with your 'real problem', break it into subproblems, break those subproblems into further subproblems, etc, all the way until you hit that atomic level that you have a solution for. DP, by contrast, is bottom up; it starts at the atomic level you have a pre-determined solution for, and uses the recurrence relations to keep building out a solution one level higher, then another level higher, until you have the answer for the level you need.

The recurrence relations for the DP solution of the Knapsack problem is as follows:

Let's define `O(k,j)` as the optimal value we can get for a knapsack of capacity `k` and with `j` items to consider. 

- $O(k,0) = 0$ for all $k$, because we get no value from no items.
- Now assume, while trying to solve for any $O(k,j)$, we had the solution to $O(k, j-1)$. Then we would just need to find the increase in our optimal value for adding that `j`th item...

The value of $O(k,j)$ as a function of $O(k,j-1)$ is:
 - $O(k,j) = O(k, j-1)$ if $w_j > k$ (if the j'th item doesn't fit in the knapsack, the optimal value is the same as for previous j-1 items)
 - $O(k,j) = max(O(k,j-1), vj + O(k-wj,j-1))$ otherwise (if $w_j \leq k$ 
    - this is less intuitive but see [this lecture](https://www.coursera.org/learn/discrete-optimization/lecture/wFFdN/knapsack-4-dynamic-programming) for more explanation

If all this seems complicated, I think DP/Bellman Equations are much easier to grok when the recurrence relations contain only a single variable; the canonical example seems to be using DP or recursion to find the n'th item in the fibonacci sequence, so if you're still confused google that for a gentler introduction to DP. The DP lecture link above from this Discrete Opt course ([also here](https://www.coursera.org/learn/discrete-optimization/lecture/wFFdN/knapsack-4-dynamic-programming)) is helpful, although I do think from about 7:00 on is much more intuitive and easier to follow than the first 7:00. So anyway if you follow that link, don't give up in the first 7:00!

Anyway, onto the code for the DP solution...


```python
def dp_select(values, weights, capacity):
    """Use dynamic programming to build up to solution for all items"""
    
    # convert params to n,K notation as used in equations in dynamic programming notes
    n = len(values)
    K = capacity
    
    # calculate table of optimal value by j,k (j items in 0..n, k is all capacities in 0..K)
    # see 9:00 - 12:30 here: https://www.coursera.org/learn/discrete-optimization/lecture/wFFdN/knapsack-4-dynamic-programming
    values_table = np.zeros((K+1,n+1), dtype=np.uint32)
    
    print("building DP optimal-value table for n,K: ", n, ",", K)
    for j in tqdm(range(1, n+1)):
        item_weight = weights[j-1]
        item_value  = values[j-1]
        for k in range(1, K+1):
            if item_weight > k:
                values_table[k,j] = values_table[k, j-1]
            else:
                values_table[k,j] = max(values_table[k, j-1], item_value + values_table[k-item_weight, j-1])
    optimal_value = values_table[-1, -1]
    print(f"optimal value is {optimal_value}. Now proceeding to derive final item-set")

    # from this table of optimal values, we now need to derive final item-set for optimal solution
    # logic of code below explained 12:30 - 14:00 at https://www.coursera.org/learn/discrete-optimization/lecture/wFFdN/knapsack-4-dynamic-programming
    taken = [0] * len(values)
    k = K   # in keeping w/ eqs, K is total capacity but k is k'th row as we move through j,k table
    for j in range(n, 0, -1):
        if values_table[k,j] != values_table[k,j-1]:
            taken[j-1] = 1
            k = k - weights[j-1]
    
    return optimal_value, taken


def solve_it(input_data):
    
    # parse the input
    (values, weights, K) = parse_data(input_data)
    
    # value-dense greedy algorithm for filling the knapsack
    obj, item_selections = dp_select(values, weights, K)

    # prepare the solution in the specified output format
    PROVED_OPTIMAL = 1
    output_data = f"{obj} {PROVED_OPTIMAL}\n"
    output_data += ' '.join(str(i) for i in item_selections)
    return output_data

```

Again, if you'd rather see it as a diff from the previous code, [check here](https://github.com/MaxPowerWasTaken/discrete_opt_coursera/commit/2e25e7ce4eb9ab044fdaa42c53680bd2f667521c#diff-57214dafea7890eb954c73d3f58b3d17)

So how'd the DP solution do?
- For Problem 1, with n=30 &  K=100,000: Optimal Solution found in 3 seconds
- For Problem 2, with n=50 &  K=341,045: Optimal Solution found in 32 seconds
- For Problem 3, with n=200 & K=100,000: Optimal Solution found in 24 seconds 
- For Problem 4, with n=400 & K=9,486,367...

{{<fig src="/img/Screenshot DP resource intensive.png"
       link="/img/Screenshot DP resource intensive.png"
       caption="But does it scale? (no)">}}

Ok so DP helpfully gives us the optimal solution on questions 1-3 (10/10 points * 3!), but the compute requirements scale exponentially. Our [tqdm](https://github.com/tqdm/tqdm) loop timer shows that for Problem 4, which is substantially larger than the first 3, the solution will likely take about 3 hours, and is maxing out my RAM plus even consuming swap. Let's try and improve this solution. 

## Incorporating Numba Just-In-Time Compilation
Per [its website](http://numba.pydata.org/):

"Numba translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library. Numba-compiled numerical algorithms in Python can approach the speeds of C or FORTRAN.

You don't need to replace the Python interpreter, run a separate compilation step, or even have a C/C++ compiler installed. Just apply one of the Numba decorators to your Python function, and Numba does the rest. 

A couple things that I found I had to tweak to get the DP solution above compiled with numba:
- numba supports limited print() functionality; "[only numbers and strings; no file or sep argument](http://numba.pydata.org/numba-doc/latest/reference/pysupported.html#built-in-functions)", which also apparently means no formatted-strings
- tqdm apparently doesn't work within numba, so I replaced it with a few more lines for progress-tracking and timinng.

I also got a deprecation warning about "reflected lists" not being supported in the future, in reference to my `weights` and `values` lists, so I just changed those to numpy arrays. All in all, I really changed very little to incorporate numba just-in-time compilation; [check out the diff here](https://github.com/MaxPowerWasTaken/discrete_opt_coursera/commit/beeb22967fdb193ad53c834bd0f26ffef1ecfea5#diff-57214dafea7890eb954c73d3f58b3d17R5)

And how did it improve performance? Re-running that same problem-4 dataset again that was on pace to take around 3 hours before...
```bash
(knapenv) maxepstein@pop-os:~/discrete_opt/knapsack$ python solver.py data/ks_400_0
building DP optimal-value table for n,K:  400 , 9486367
Tracking Progress. j(n) = ....
-------------------
20
40
60
80
100
120
140
160
180
200
220
240
260
280
300
320
340
360
380
400
completed dp_select() in 0 days 00:01:35.641159
3967180 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

...just to be clear, that "0 days 00:01:35.641159" is 1 minute and 35 seconds. That's a truly wild speedup for such a simple change. Really really impressed with numba. Problem 5 actually completes even faster. Problem 6, with `n, K` = `10,000 1,000,000` results in: 

```bash
(knapenv) maxepstein@pop-os:~/discrete_opt/knapsack$ python solver.py data/ks_10000_0
Segmentation fault (core dumped)
```

## Wrapping Up, Final Thoughts (for now)
- A Dynamic Programming(DP) implementation in Python found the optimal value quickly for the first 3 problems, but would have taken hours for the 4th problem and crashed on 5 & 6.
- Wrapping the DP function with a numba `@njit()` decorator, and very little additional modification, yielded the optimal solutions for Problems 4 & 5 in about 2 minutes combined. It still crashed for Problem 6 (the largest problem)
  - My final solution at this point scores 53/60 points - DP (optimal) solution for problems 1-5 (5 * 10/10), and value-dense greedy for problem 6 (3/10)
  - I also tried calling `dp_select` on the largest subset of Problem 6 I could portion off without crashing my laptop, and then using the value-dense greedy selection for the remainder of the problem, but that didn't improve my score. 

So where to go from here?
- I tried using a declarative constraint programming language(CP) called Minizinc. There's a lot that's cool about Minizinc, but ultimately it performed much less well than my existing Python solution. But will probably revisit Minizinc / CP later.
- I tried using [Google OR-tools' Knapsack-Problem Solver](https://developers.google.com/optimization/bin/knapsack) with Python; that handled this programming assignment's largest problem easily. But that kinda feels like cheating, or at least defeating the purpose of the class, for use in this assignment. But it's definitely worth knowing that tool exists, for any future problem I encounter that can be clearly mapped to the knapsack problem.
- A Branch & Bound approach, or other local-search algorithm, I think would be promising here. Branch & Bound was covered a bit in this week's (week 2) lecture, but week 4 is all about local-search algorithms. 

The professor for this course suggests not pressing too much on getting a 60/60 on each assignment on the first try, because while going through the course, you learn additional techniques   

