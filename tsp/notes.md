+++
Week 4 - Local Search
+++

## Lecture 1 - Intuition, n-queens
What is it?
- Move from one configuration to another similar one; a "local move"
- start from a complete assignment to the decision variables; then modify >=1 of them
    - contrast with Constraint Programming, where we start with assignments to a subset of decision variables and then extend them


Local Search for Satisfaction Problems
- transform the problem into an optimization problem
    - use the concept of violations.
        - i.e. how many constraints are violated by a configuration
        - try to move towards a configuration with fewer violations
        - can also measure violations  instead of as 0/1 violated or not, but as degree of violation (e.g differentiate between n=16 and n=30 if constraint is n<15)

What is a local move?
- many choices.
    - simplest case: pick a decision variable and change it's value

How to select a move?
- many choices
- max/min conflict
    - Choose a variable which appears in a large number of violations
    - Change the value of that variable to resolve as many of those constraints as possible

Key Concept: Neighborhood
- A neighborhood is the set of configurations you can get to from a given configuration, within 1(?) local move
- Local Search will find you a local minimum, and a local optimum is defined with respect to a neighborhood. 
    - No guarantees that it will be a global optimum. Can be arbitrarily good or arbitrarily bad

## Lecture 2 - Swaps, car sequencing, magic square.
- Sometimes, insted of changing one decision variable at a time, it's good to swap values of two variables
    - this is useful when you have some sort of global constraint you can maintain satisfaction of by only swapping decision-variable values in pairs


sometimes worth distinguishing between:
- hard constraints: always kept feasible during search
- soft constraints: may be violated during the search

## Lecture 3 - Optimization, 2-opt/k-opt
Local Search for Optimization

Consider Warehouse siting problem
- Given a set of possible-warehouses and (actual) customers, each with an associated location...
- Decide which warehouses to open, such that...
- Minimize combined fixed cost of all open warehouses, plus transportation costs (distance between each customer and nearest warehouse)

So what would we call a neighborhood?
- Simplest neighborhood: open and close warehouses (flip the value of $O_w$ 0->1 or 1->0)
- Union of neighborhoods
    - open and close a warehouse (swapping two warehouses)

*Traveling Salesman*
Given
- A set $C$ of cities to visit
- A symmetric distance matrix $d$ between every two cities

Find
- a tour of minimal cost visiting each city exactly once

- Probably the most studied combinatorial problem.

A CSP Pseudocode for TSP:
```
range cities = 1..n;
int distance[cities, cities] = ...;
var{int} next[cities] in cities;

minimize sum(c in cities) d[c, next[c]]

subject to circuit(next);
```

Note: Minizinc does have a circuit constraint:
```
predicate circuit(array [int] of var int: x)

Constrains the elements of x to define a circuit where x [ i ] = j means that j is the successor
of i .
```
source: https://www.minizinc.org/doc-2.3.0/en/MiniZinc%20Handbook.pdf


What is going to be the neighborhood for the TSP?
- Lots of possibilities
- 2-opt: the neighborhood is the set of all tours that can be reached by swapping two edges
    - select two edges and replace them by two other edges
- 3-opt
    - much better than 2-opt in quality but more expensive
- 4-opt is often marginally better in quality but much more expensive
- k-opt


## Lecture 4 - Optimality vs Feasibility, graph coloring
- Two aspects
    - optimization: reduce the number of colors
- Feasibility: two adjacent vertices must be colored differently.

How to combine them in local search?
- Three ways
1. Reduce to a sequence of feasibility problems
2. Stay in the space of feasible solutions, only look at legal colorings
3. Consider feasible and infeasible configurations,
    - says this is useful in graph coloring

*Approach 1: Reducing to a Sequence of Feasibility Problems*
Steps:
1. find an initial solution with k colors
    - e.g. greedy algorithms
2. Remove one color (k'th color)
3. randomly assign all nodes colored with k to a remaining color in 1..k-1
4. find a feasible solution with the k-1 coloring 
    - e.g. by the process outlined before of: 
        A) evaluating constraint violations
        B) select decision variable involved in most violations (e.g. vertex(node) connected to most same-colored nodes)
        C) change its value to whatever would cause the fewest violations by that node (change node's color to least-common color among all nodes its connected to) 

*Approach 2: Explore only in the Space of Feasible Solutions* 
Note: I don't like this approach for the graph coloring problem at least.
- Say you considered a local move 'change the color of a vertex.' Given 

- Neighborhood
    - Change the color of a vertex
- Objective Function
    - minimizing the number of colors
- How to guide the search?
    - Changing the color of a vertex typicaly does not change the number of colors

Change of Perspective: Color Classes
- $C_i$ is the set of vertices colored w/ i
- Use a proxy as objective function
- Favor large color classes
    - The (indirect) objective function becomes $maximize \sum_{i=1..n} |C_i|^2$
- But can be nonobvious how to come up with a neighborhood of local moves that never violate the constraint of graph coloring.
  - Enter 'Kemp Chains'
  - say you have two color classes, $C_i$ & $C_j$
  - you can take all the vertices in each that are connected to each other, and swap their colors. no new violations, might make one color class larger (and the other smaller)
  - I don't see how this extends to be safe and keep from adding violations if you have e.g. 10 colors though and each node is connected to many different nodes of many colors. seems to me like it might then introduce infeasible configs. but maybe I'm missing something.

*Approach 3: Exploring both Feasible and Infeasible Colorings*
- The search must focus on reducing the number of colors and ensuring feasibility
- How to combine optimization and feasbility?
  - Make sure that local optimums are feasible
  - use an objective function that balances feasibility and optimality
  - e.g. $min \ w_f f + w_o O$

So what's a neighborhood here?
- Simple neighborhood: change color of a vertex

Bad edges
- Just a constraint violation in graph coloring: two vertices with same color are linked by a 'bad edge'
- $B_i$ is the set of bad edges between vertices colored with color $i$.

Wrapping up Approach 3: 
So what does $min \ w_f f + w_o O$ look like for this problem?

*Decrease number of colors*
 using the large-color-classes heuristic:
$max \sum_{i=1..n} |C_i|^2$ (presumably n = n-colors)

*Remove violations*:
minimize number of bad edges:
$min \sum_{i=1..n} |B_i|$

Then he says for graph coloring specifically, if you put these together in a slightly tweaked way, you get an objective function with the very convenient property that a local minimum is guaranteed to be a feasible coloring. So that 'magic' objective function for this problem is:

$min \sum_{i=1..n} 2*|B_i|*|C_i| - \sum_{i=1..n} |C_i|^2$

(the minus in the second term is because we want to maximize that term, but we're sticking it in an obj function we're minimizing)

My general conclusion on this:
- Going back to graph coloring, I'd be most interested in using Approach 1 because it's so clear to me from general principles (e.g. it's explanation from n-queens) how it applies to graph coloring and why it's generally useful. But might also be worth implementing this approach 3 as well. Approach 2 seems poorly suited to graph coloring. But probably good strategy for TSP to always keep configs feasible.

## Lecture 5: Complex Neighborhoods, Sports scheduling
Hard to take notes. Talked about different local moves (neighborhoods) for the Tournament Scheduling Problem, a generalization of MLB scheduling. But was pretty hard to follow. Kind of boring actually.

## Lecture 6 - Escaping Local Minima, Connectivity
Connectivity represents if you can get to the optimal solution from a particular configuration from only a series of local moves.

Doesn't mean your local solver *will* find that optimum, because may require a series of temporarily loss-increasing moves on the way to that optimum. But still better than being in a region that's not connected (although presumably random restarts could also help you with that)

## Lecture 7 - Formalization, heuristics, meta-heuristics
General LS algorithm:
First some nomenclature...
- $s$ 
    - States: either solutions or configurations
- $N(s)$ 
    - Neighborhood of $s$; Moving from state $s$ to one of its neighbors 
- $L(N(s),s)$
    - Set of legal neighbors (some are legal, others are not)
    - (this is not necessarily 'feasibility'. Just some set of criteria you've baked into your LS process to filter out which types of moves you want to allow)
- $S(L(N(s),s),s)$
    - selection function for selecting which legal neighbor to move to.
    - many ways to do this.
- final equation above looks complicated, but all it really says is *select* a *legal* *neighbor*. 
- $f(s)
    - objective function we're **minimizing**

### Formal Pseudocode Algorithm
```
function LocalSearch(f, N, L, S):
    s = GenerateInitialSolution();
    s* = s;  # s* is best solution so far
    for k in 1 to MaxTrials:
        if satisfiable(s) & f(s) < f(s*):  # to minimize f
            s* = s
        s = S(L(N(s),s),s);
    return s*;
```

So what would be a set of legal moves?
- e.g. all local improvements - all n in N such that f(n) < f(s)

What would be a selection function?
- e.g. greedy - S(L,s) = argmin(n in L)f(n)

Heuristics
- how to choose the next neighbor
- use local information
    - the state s and its neighborhood
- goal is to drive the search towards a local minimum

Metaheuristics
- aim at escaping local minima
- drive the search towards a global minimum
- typicall include some memory or learning

Selection Functions
- Randomization often important (more on this later)
- Best neighbor: return state (from legal neighbors) with min f(s)
    - alternatively identify several best-neighbors and return one of them randomly
- Best improvement: hm I don't get how this is different. isn't the best neighbor also the best improvement from your current state?

Multi-State Heuristics
Motivation:
- Avoid scanning the entire neighborhood
- Still keep a greedy flavor
2 approaches:
- Max/Min conflict from before
    - Stage 1: select variable with most violations (greedy)
    - Stage 2: select value w/ fewest resulting violations (greedy)
    - Already saving significant time vs scanning result of scanning every possible move for every possible queen by narrowing evaluation to one queen in stage 1

- Min-conflict heuristic
    - Like Max/Min conflict but randomly select variable (e.g. which queen) (albeit a var w/ at least one violation) to modify.
    - So first stage randomized, second stage like before

Says because of complex interactions between variables, sometimes the randomized first stage is better. Not necessarily the queen which itself is involved in most violations is the one which changing would reduce total violations by the most. Sometimes the randomized first stage is better.

Algorithmic Complexity (Big-O notation)
- naive neighborhood approach - scanning each possible value for each possible queen would take quadratic time 
    - $O(n^2)$ where $n$ = number of queens.
- Min-Conflict is $O(n)$
    - constant time for random lookup + linear time for var selection
- Max/Min Conflict also $O(n)$. Slower than min-conflict but still linear time.
    - linear time for max-violation lookup + linear time for var selection

Third approach
- Random Walk: select a neighbor completely at random
  - Second stage: decide whether to accept it
    - like if it's an improvement
    - metropolis algorithm: 

## Lecture 8 - iterated local search, metropolis search, simulated annealing, tabu search intuition

How to escape local minima? Approaches:
- Iterated local search: execute multiple local search from different starting configuration.
    - Each will lead to a local optimum
    - At the end just choose the best local optimum
    - Also called LS "w/ restarts"
    - Can be combined with many other metaheuristics

- Metropolis Heuristic (simulated annealing is based on this)
    - accept a move if it improves the object value, or in case it does not, with some probability
    - the probability depends on how "bad" the move is $(f(n) - f(s))$
        - also based on a temperature parameter $t$
        - degrading move is accepted with probability:
            - $exp(-\Delta / t)$
            - very large $t$ means you're basically a random walk
            - very small $t$ means almost never accepting a degrading move; "more greedy" 
    - inspired by statistical physics
- Simulated Annealing
    - Metropolis strategy above, w/ proviso that:
        - start with a high temperature, decrease temperature over time
        - reminiscent of RL - start with more exploration, focus down on greedier / more exploitation later on
Additional Techniques used with Simulated Annealing
 - Restarts
 - Reheats: depending on results, maybe you're cooling too fast so jack the temperature back up
 - tabu search

 Tabu Search:
 - Goal is sorta like restart, except instead of restart from random location, once you hit a local minimum, you're trying to jump out of it to a nearby space where you might continue on to a different (better) local minimum.

 - Keep track of nodes I have already visited: the 'tabu list'
    - don't want to revisit these
    - "legal moves are not tabu"

Many Other Metaheuristics
- variable neighborhood search
- guided local search
- ant-colony optimization
- hybrid evolutionary algorithms
- scatter search
- reactive search
- many others...

## Lecture 9 - Tabu Search Formalized, Aspiration
Key ideas:
- Maintain list of configurations already visited (tabu list)
- Try to visit best configuration that is not tabu
- Key issue:
    - expensive to maintain list of all visited nodes
    - and every time you select a neighbor, you have to spend time checking if it's been visited already
    - my question? hash maps help here?
- Solution: Short-term memory
    - just keep recently visited nodes in your tabu list
    - can increase/decrease the size of your tabu list dynamically
        - decrease when the selected node degrades the objective
        - incrase when the selected node improves objective
- But: Still costly to store entire configurations of even a subset
    - Solution: keep a concise abstraction of these configurations
    - Many possibilities
        - one option: store transitions (e.g. swaps like (a1, a17)) instead of whole configuration
        - a configuration is tabu in this case if it can be obtained by swapping a pair in this transition-tabu list
        - I'm confused how this really gets you what tabu search promises. Banning a particular swap for e.g. 100 moves seems like a bad proxy for avoiding returning to a particular state? A particular swap from one state does not take you to the same state if you make it from a different state? Why not just hash the whole configuration? 
        - Ok Prof touches on this...
Transition Abstractions in Tabu List - Too Strong or Too Weak?
- A bit of both
- Too weak in that they cannot prevent cycling (visiting previously-visited configurations) since they only contain partial-info about a configuration
- Too strong:
    - They may prevent us from visiting configurations that have not yet been visited
    - The swaps would produce different configurations (from a new initial state) but tehy are forbidden by the tabu list 

- Aspiration Criterion
    - If a move is tabu but looks sufficiently good, you take it anyway

Long Term Memory: A More Appealing (to me) Metaheuristic
- Intensification: store high quality solutions. Return to them periodically
    - e.g. after hitting (getting stuck in) a local optimum. If there's some randomness to your search, won't necessarily repeat prev path to current local optimum.

Diversification: 
- When the search is not producing improvement, diversity the current state
    - e.g. randomly change the values of some variables

Strategic Oscillatoin
- Change the percentage of time spent in the feasible and infeasible regions

Final Remark for this week:
- These techniques such as divesification, intensification and strategic oscillation are useful in many different settings, e.g. simualated annealing and many other metaheuristics. Use those in those situations as well.

Notes on Traveling Salesman Problem
- First thought as he shows a cloud of points
    - Could break it down into subparts:
    - e.g. "the points on the left, solve that, and the points on the right, solve that, and then maybe there's some obvious way of linking those two"

- He gives a formulation of the problem (basically same as in assignment pdf) but notes "don't assume the formulation we give you is the best formulation of the problem" and that different formulations of the same problem may work better/worse for different technologies.

- He suggests visualizing really helpful for this assignment, and edges crossing each other is always a red flag that your tour can be improved.

Other tips for this assignment
- FAST neighborhood calculation
- Symmetries - there are plenty in this problem
- Do you need every edge?
    - In the full distance matrix, every city is/can be connected to every other city. Do you really need (to consider) all of those?
- Coomplete search / Lower bounds (if you're doing branch-and-bound for complete search, think about how you're setting lower bounds. many ways to do so here).
- Look at the solution - visualize as said above
    - https://discreteoptimization.github.io/

- from googling - use a solver as a integer linear program?

NOTES ON LOCAL-SEARCH / TSP BLOG...
- greedy starting at node0 got 3/10 for each: 15/60 (memory error on prob6)
- trying greedy but starting at every different node increases score to 7/10 on one of them - 19/60. basically swing and a miss 
- 