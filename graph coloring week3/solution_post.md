## Constraint Programming & The Graph Coloring Problem 
Week 3 of [Discrete Optimization](https://www.coursera.org/learn/discrete-optimization/home/week/3) on Coursera covered Constraint Programming, and the programming assignment is the Graph Coloring Problem.

Constraint Programming(CP) seems to me to be a very elegant approach. You start by describing the structure of a problem, by coding:
- parameters, 
- decision variables, 
- constraints, and 
- an objective function. 

This problem description can be translated to, by the Constraint Programming language, a certain space of possible solutions.

The CP language will first use the constraints given to eliminate as many theoretical solutions as possible; i.e. to prune the solution space. It will then select a possible value for one of the decision variables, re-check the constraints to again further prune the solution space, rinse and repeat. Often it will encounter that the set of decision variable values it has assigned so far has led it to an infeasibility, in which case it will backtrack to a previous decision, select a new possible value for that decision variable, re-assess the remaining and pruned search space, and proceed again. A great description of this process with some really helpful diagrams for how a CP reasons through the n-queens problem is available at [Section 2.5.1 Finite Domain Search of the Minizinc documentation](https://www.minizinc.org/doc-2.3.1/en/mzn_search.html)

Because we need constraints to prune the solution space, adding more constraints to our problem description can be really helpful. 

### The Graph Coloring Problem 
We are given an undirected graph; a set of nodes and edges where each edge connects two nodes, and each node has a  color. The problem is to find the coloring of the graph (the color assigned to each node) that:
 - minimizes the number of distinct colors used,
 - such that no two nodes connected by an edge have the same color.

Formally, it's specified as:
- Given a graph  $G = <N,E>$ with nodes $N = 0..n-1$ and edges $E$, let $ c_i \in \mathbb{N}$ denodes the color of node $i$...

minimize: $max(c_i)_{i \in 0..n-1}$
subject to: $c_i \neq c_j$ for all $(<i,j> \in E)$

### Data Format Specification
- Our input files have $E$+1 lines:
  - First line is $|N| |E|$ (number of nodes & number of edges)
  - Each following line has two numbers. They represent two nodes, which define an edge

### Diving into some code 
I figured this would be a good time to try out the Constraint Programming language [Minizinc](https://www.minizinc.org/) some more, which I mentioned briefly at the end of my [knapsack blog post](https://maxpowerwastaken.gitlab.io/model-idiot/posts/knapsack_blog/). Minizinc allows you to express mathematical optimization & constraint-feasilibity/satisfaction problems in a nice high-level language (though not a 'general purpose' language), which Minizinc then compiles down to a common standard called Flatzinc. Since many solvers can solve Flatzinc models, Minizinc can be run with (and provides a common interface/language to) a variety of backend solvers, both opensource and commercial.


I'll start by working with the smallest input data file we're given:
```
maxepstein@pop-os:~/discrete_opt/graph coloring week3$ cat data/gc_4_1
4 3
0 1
1 2
1 3
```
Ok so we have 4 nodes, 3 edges, and our edges go:
- From node 0 to node 1
- From node 1 to node 2
- From node 1 to node 3

Here's how we can describe this graph coloring problem in Minizinc. You'll see the code mirrors the description of Constraint Programming I mentioned above, with the following four components:
- parameters, 
- decision variables, 
- constraints, and 
- an objective function. 

```minizinc 
% Parameters: we need to specify values for these (input data)
int: NUM_NODES = 4; 
int: NUM_EDGES = 3;
array[1..NUM_EDGES,1..2] of int: edges = [|
0, 1|    
1, 2|      
1, 3|];

% Decision Variable: the solver will find the values of these
array[0..NUM_NODES-1] of var 1..NUM_NODES: color;

% Our Constraints 
constraint forall(e in 1..NUM_EDGES)(color[edges[e,1]] != color[edges[e,2]]);

% Our Objective Function
solve minimize max(color);
```

We can run this in the Minizinc IDE, or from the shell:
```bash
~/.../$ minizinc graph_coloring.mzn
color = array1d(0..3, [2, 1, 2, 2]);
----------
==========
```
Ok so the model finds the optimal solution, using two colors. Color '2' for the first, third and fourth node, and color '1' for the second.

### First Reactions to our MiniZinc Code
#### So Declarative Languages are Cool
This to me is wild. Really super cool. Yeah this was a tiny problem size, but all I did was write a description of the problem - none of the code above really deals with *how* to solve the problem at all. And we get a solution which is proved optimal.  

And while the syntax for Minizinc is unfamiliar to me (and may look so to you), it does seem intuitive. In fact it's very similar to how I would actually write down a description of this problem by hand in a notebook. Even though I'm generally most comfortable in a Python environment, I actually like the look of Minizinc code much more than what I tend to see for mathematical programming in Python, like [PuLP](https://github.com/coin-or/pulp) for example.

So that's it? We describe the problem and we're just done? Well not really. Like before, as we scale the size of the dataset (based on my limited experience with Minizinc) we'll still run into some issues. And then what do we do? Well, there are many options apparently, and apparently how to improve performance of a Minizinc model is bit of both an art and science. Read on for what I've learned so far, over the course of using Minizinc .


#### A Note on Array Indices & Values
Notice in our solution output above ( `color = array1d(0..3, [2, 1, 2, 2]);` ), `array1d(0..3` refers to the fact that the `color` array is *indexed* on the sequence `0,1,2,3`, not that it can take `values`  from `0` to `3`. The values it can take are `[1,2,3]`.

This is due to how we declared our `color` array above: 
`array[0..NUM_NODES-1] of var 1..NUM_NODES: color;` 
The range in the brackets is what you are indexing the array on, the range after `of` is what values it can take. This can be confusing at first (or second).

We could change the code above to change the `values` of colors to be zero-indexed (if that seems more consistent), but we could not change the array's `index` to be 1-indexed without throwing an error. The reason is that the index of our color-array is really our set of nodes; we need to find one color per node, and in the datasets we're given, we always have a Node 0.

### Running our MiniZinc Model on the Assigned Problems
Actually first, a quick sidebar:
#### A Convenience Shortcut for this Class
In this class we're always given many data files to test our solver on, and the `submit.py` script will always run six of them and submit our solutions for grading. By looking through this `submit.py` script, we can see that the problems which are graded are selected from a file in the directory we're given called `_coursera`. If we cat that we see:
```bash
(colorenv) ~/.../$ cat _coursera
wNBw6FwlEeaEFQ4KWsLmjw
Graph Coloring
fmYLC, ./data/gc_50_3, solver.py, Coloring Problem 1
IkKpq, ./data/gc_70_7, solver.py, Coloring Problem 2
pZOjO, ./data/gc_100_5, solver.py, Coloring Problem 3
XDQ31, ./data/gc_250_9, solver.py, Coloring Problem 4
w7hAO, ./data/gc_500_1, solver.py, Coloring Problem 5
tthbm, ./data/gc_1000_5, solver.py, Coloring Problem 6
```
So now we can set handy aliases to reference the specific datafiles (from the 35 or so we're given) we need to solve for the programming assignment:
```bash
(colorenv) ~/.../$ prob1=data/gc_50_3
(colorenv) ~/.../$ prob2=data/gc_70_7
(colorenv) ~/.../$ prob3=data/gc_100_5
(colorenv) ~/.../$ prob4=data/gc_250_9
(colorenv) ~/.../$ prob5=data/gc_500_1
(colorenv) ~/.../$ prob6=data/gc_1000_5
```

#### Back to Running Our MiniZinc Model on the Assigned Problems
So we can check out the scale of the graph problems we have for this week:
```bash
(colorenv) ~/.../$ head -1 $prob1
50 350
(colorenv) ~/.../$ head -1 $prob6
1000 249482
```
Ok cool. so prob1 has 50 nodes and 350 edges, and prob6 has 1,000 nodes and 249,482 edges. Hokay... back to code...

In the Minizinc code above, I added the data from one of our given data files to the Minizinc code, and formatted it to be proper Minizinc syntax. Moving forward, to run this model on arbitrary data files, we need to take two steps:
- Separate the modeling code from the data-input code 
    - This can be done by moving the assignment statements for our parameters into a `.dzn` file.
- Write a script that takes the data input files we're given, and reformats them to suitable `.dzn` data files for the minizic solver.

##### Separating Model From Data in Minizinc
We can break out the `.mzn` minizinc file above into the following two files:

`graph_coloring.mzn:`
```minizinc
% Input Parameters (values for these specified in .dzn data file)
int: NUM_NODES; 
int: NUM_EDGES;
array[1..NUM_EDGES,1..2] of int: edges;

% Decision Variable: the solver will find the values of these
array[0..NUM_NODES-1] of var 1..NUM_NODES: color;

% Our Constraints 
constraint forall(e in 1..NUM_EDGES)(color[edges[e,1]] != color[edges[e,2]]);

% Our Objective Function
solve minimize max(color);

% formatted output
output[show(color)];
```

`mzn_data/gc_4_1.dzn`:
```minizinc
NUM_NODES = 4; 
NUM_EDGES = 3;
edges = [|
0, 1|
1, 2|
1, 3|];
```
Now, although the model file still instantiates the parameters, it doesn't assign any particular values to them, so this model can be run with different data files swapped out. Also I added an `output` statement which formats the `color` array a bit.

Now for a python function that can turn our input data files which look like this:
``` bash 
maxepstein@pop-os:~/discrete_opt/graph coloring week3$ cat data/gc_4_1 
4 3
0 1
1 2
1 3
```

into this:
```bash
maxepstein@pop-os:~/discrete_opt/graph coloring week3$ cat mzn_data/gc_4_1.dzn 
NUM_NODES = 4; 
NUM_EDGES = 3;
edges = [|
0, 1|
1, 2|
1, 3|];
``` 
...
```python
def convert_data_to_dzn(input_data, dzn_filename = 'data.dzn'):
    '''Reformat input_data to a minizinc .dzn file, write to disk'''
    
    # parse the input
    lines = input_data.split('\n')
    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    
    with open(f'mzn_data/{dzn_filename}', 'w') as f:
        f.write(f"NUM_NODES = {node_count};\n")
        f.write(f"NUM_EDGES = {edge_count};\n")
        f.write(f"edges = [|")
        for line in lines[1:-1]:  # skipping last line which is blank in data files
            node1, node2 = line.split()
            f.write(f"\n{node1}, {node2}|")
        f.write(']\n')
    
    return None
```

So here's our first `solver.py`, which basically:
- reformats the input datafile to a .dzn minizinc data file using `convert_data_to_dzn` defined above
- calls our minizinc model
- reformats the output per the coursera course requirement

`solver.py`:
```python
import subprocess
from convert_data_to_dzn import convert_data_to_dzn


def solve_it(input_data):

    # Run input_data through Minizinc Model
    mz_model = "graph_coloring.mzn" 
    convert_data_to_dzn(input_data, dzn_file = f'{orig_filename}.dzn')

    start = pd.Timestamp.utcnow()
    mz_output = subprocess.run(["minizinc", mz_model, f"{orig_filename}.dzn"], 
                                shell=False, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(f"finished minizinc run in {pd.Timestamp.utcnow()-start}")

    # Parse Minizinc Output and Prepare Coursera Course Output
    OPTIMAL = 0
    if "==========" in mz_output:
        OPTIMAL = 1

    colors = mz_output.split('\n')[0].strip('[]').split(', ')
    num_colors = len(set(colors))

    output_data = f"{num_colors} {OPTIMAL}\n"
    output_data += ' '.join(colors)

    return output_data
```
And to confirm it works:
```bash
(colorenv)~/.../$ python solver.py data/gc_4_1
2 1
2 1 2 2
```
Sweet. Onto the graded problems:
```bash
(colorenv)~/.../$ python solver.py $prob1
6 1
2 6 2 4 4 5 6 1 6 3 4 1 6 1 2 5 1 3 4 6 2 4 2 5 5 2 5 3 5 4 6 3 4 2 3 2 6 3 2 1 6 4 1 1 6 1 4 5 4 3
```

Cool. Optimal solution and 10/10 on Problem 1.

However, Problem 2, with 70 nodes & 1,678 edges hangs for a while (at least a few minutes before I killed it).


### Improving our Minizinc Model
There are several different ways to try to improve performance of a Minizinc model. A (non-exhaustive) list includes:
- try running with a different built-in backend solver
  - running `minizinc --solvers` will show which ones are available
- install a non-built-in backend solver and try that
  - for example Google's OR-Tools is apparently a possible backend
- try some of the compilation optimizations shown under `Flattener two-pass options` section of `$ minizinc --help`

...but as far as I can tell the first thing to try is generally to add additional constraints which will helpully restrict the search space. 

#### Symmetry Breaking
How might we want to restrict our search space? A big topic that came up in the lectures was *symmetry breaking*. *Symmetry* here means that many regions of our initial search space are essentially equivalent, and will each yield equivalent solutions. We can arrive at the same optimal solution much faster if we add a constraint to avoid searching through multiple *symmetric*, or essentially equivalent, regions of the search space.

To show what I mean, let's run our previous minizinc model, but with the following slight adjustments:
Replace `solve minimize max(color);` with:
```
constraint max(x) = 2;  % we know optimal solution for gc_4_1 dataset uses 2 colors
solve satisfy;          %  find any solution that satisfies all constraints
```

We can now run this model and get *all* solutions that satisfy the consraints with:
```bash
(colorenv) ~/.../$ minizinc graph_coloring_scratch.mzn mzn_data/gc_4_1.dzn --all-solutions

color = array1d(0..3, [2, 1, 2, 2]);
----------
color = array1d(0..3, [1, 2, 1, 1]);
----------
==========
```
...This should help illuminate symmetry. The two solutions found above are, for practical purposes, the same solution. They each mean:
- Two colors suffice to color the graph
- Use one color for the second node; use another color for the other three nodes

Our question for symmetry-breaking is what constraint could we add that would ensure we'd find one of those solutions (even once we remove our `max(color) = []` constraint, which we wouldn't know in advance on new problems), without further exploring the solution space on the way to additional symmetric solutions?

The first thing that comes to mind is `constraint color[0] == 1`. That works for this problem, but helps less for larger problems with more disinct colors needed.

Luckily, Minizinc has a cool built-in constraint called `value_precede`, with [the following signature](https://www.minizinc.org/doc-2.3.0/en/lib-globals.html):

```minizinc

predicate value_precede(int: s, int: t, array [int] of var int: x)

Requires that s precede t in the array x
```
This will let us ensure that, if e.g. 4 colors are needed to color a graph, Minizinc will stop at [1,2,3,4] and not keep searching for symmetric (equivalent) solutions like `[2,1,3,4], [1,3,2,4]`, etc. (PS Thanks to github user Kim0 and [his graph-coloring minizinc solution](https://github.com/kim0/graph-color-minizinc/blob/master/solver.mzn) for introducing me to `value_precede`)

So here's our Minizinc solution with our additional symmetry-breaking constraint, which I'm calling for now `graph_colorin2.mzn`:

```minizinc
include "globals.mzn";

% Input Parameters (values for these specified in .dzn data file)
int: NUM_NODES; 
int: NUM_EDGES;
array[1..NUM_EDGES,1..2] of int: edges;

% Decision Variable: the solver will find the values of these
array[0..NUM_NODES-1] of var 1..NUM_NODES: color;

% Problem-Statement Constraint 
constraint forall(e in 1..NUM_EDGES)(color[edges[e,1]] != color[edges[e,2]]);

% Symmetry-Breaking Constraint
constraint forall(n in 1..NUM_NODES-1)(value_precede(n, n+1, color));

% Our Objective Function
solve minimize max(color);

% formatted output
output[show(color)];
```

To see the effect of this, I solved for all-solutions for the second-smallest dataset we're given (gc_20_1), with and without the `value_precede` constraint. gc_20_1 has 20 nodes, 23 edges, and optimal solution requires using 3 distinct colors.

Running with the `value_precede constraint` took 1.5 seconds. Running without it took 10.2 seconds and found 6x as many (symmetric/eqivalent) solutions.

### On to Problem 2
Dang, hanging again. Ok so much for all that about symmetry-breaking. I mean, we still know that unambiguously improved our Minizinc model, but apparently not by enough to quickly find the optimal solution to Problem 2. Bummer. 

After googling some more on Minizinc modeling, I found [this really impressive blogpost](https://www.hillelwayne.com/post/minizinc-2/). The author goes through a lot of different potential minizinc code improvements, which he finds in fits and starts, with some dramatic improvements among many regressions. The following statements in particular are what I had in mind when describing optimizing minizinc code as a bit of an art as well as a science:

"One solver’s improvement is another solver’s regression."
&

"""
1. Optimizations are nonlinear. Whether something’s an improvement or regression depends on your other optimizations.
2. Whether something is an improvement or a regression depends on the data set.

"""

After plenty of playing around with Minizinc myself, both of these observations seem painfully familiar. Still, I'm sure I'll get a better feel for how to reliably code efficient Minizinc models over time. 

### Trying Google OR-Tools' Backend Solver
Apparently Minizinc holds an annual competition for backend solvers. And for the 2019 competition, [four of the five categories were won by Google's OR-tools](https://www.minizinc.org/challenge2019/results2019.html). So let's try that.

OR-tools can be `pip install`ed for the python package, but for use with Minizinc you need to down the FlatZinc binary for your OS [from here](https://developers.google.com/optimization/install#binary). Then follow the instructions `[from here]` on creating a "model solver configuration"(msc) file for OR-tools, which also references [this other page](https://www.minizinc.org/doc-2.3.0/en/fzn-spec.html#sec-cmdline-conffiles) on where to put it.

Alright now that we have Google OR-tools, let's try it on that pesky problem 2. We'll also specify running in parallel on 8 threads since apparently OR-tools is particularly good at parallelizing FlatZinc models.

```bash
minizinc graph_coloring2.mzn $prob2.dzn -p 8 --solver org.ortools.ortools




LEFT TODO
- download google or-tools backend
- see that still doesn't help
- try constraint-satisfaction 

DONE
- introduce value_precede
- credit kim0.
- show how it reduces space of possible solutions for next-smallest problem by []X.
- also see if adding color[0]==1 leads to faster solution time.



- show value-precede constraint and describe
- show reduces 



Well we could add `constraint color[0] = 1;`. That eliminates 

NEXT: Measure number of --all-solutions found with optimal value for data/gc_20_1 ($prob1 way too big; way too many solutions even with value-precede). 
  - min max(color) on this problem leads to max-color=2 (ncolors=3)
  - w/ value_precede constraint, runs for 1.9s & finds nSolutions=41,472
  - w/o value_precede constraint, runs for 10.8s & finds nSol = 248,832
    - how many seqs of 0,1,2 in 20 places are there???
    - also that's not even 10x more? How many for no value-precede, but just color[0]==0. has to be like 1/3 right?
      - ran for 3.3s, nsolutions = 82,944

      - think I'm looking for an permutation with repetition https://en.wikipedia.org/wiki/Permutation
      - "If the set S has k elements, the number of n-tuples over S is: k^n"
      - ...so...3^20 = 
        - does that make sense? try a small example.
        - S = (a,b), n=3. 2^3 = 8...
          - a,a,a
          - a,a,b
          - a,b,a
          - a,b,b
          - b...

  ```bash
  [1, 2, 0, 1, 0, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1]
----------
==========
%%%mzn-stat: initTime=0.000281
%%%mzn-stat: solveTime=10.842
%%%mzn-stat: solutions=248,832
%%%mzn-stat: variables=20
%%%mzn-stat: propagators=24
%%%mzn-stat: propagations=175031
%%%mzn-stat: nodes=497663
%%%mzn-stat: failures=0
%%%mzn-stat: restarts=0
%%%mzn-stat: peakDepth=18
%%%mzn-stat-end
%%%mzn-stat: nSolutions=248832
```

- hm, still hangs for at least 10 minutes on problem2, even after 



#### errata
The `----------` in the output below `color...` means MiniZinc found a solution, and the `==========` means MiniZinc proved that solution is optimal.