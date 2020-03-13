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
```
...still hangs. Ok this is getting annoying.

### A Change in Perspective: Don't Let the Optimal Get in the way of the Good
Ok so recalling that a key aspect of NP-hard problems is it's *very* hard to get a (provably)(globally) optimal solution, let's just see how good the solutions are that we can get easily with our current minizinc code. Below is a slightly modified version of our previous code, where I add a `MAX_COLOR` variable, and instead of solving to `minimize(max(color))` we'll `solve satisfy max(color) <= MAX_COLOR`. Full current Minizinc code is:

```minizinc
include "globals.mzn";

% Input Parameters (values for these specified in .dzn data file)
int: NUM_NODES; 
int: NUM_EDGES;
int: MAX_COLOR;
array[1..NUM_EDGES,1..2] of int: edges;

% Decision Variable: the solver will find the values of these
array[0..NUM_NODES-1] of var 1..MAX_COLOR: color;

% Our Constraints 
constraint forall(e in 1..NUM_EDGES)(color[edges[e,1]] != color[edges[e,2]]);
constraint forall(n in 1..NUM_NODES-1)(value_precede(n, n+1, color));

% Our Objective Function
%solve minimize max(color);
constraint max(color) <= MAX_COLOR;
solve satisfy;

% formatted output
output[show(max(color)) ++ "\n" ++ show(color)];
```

Since I want to try out many different (incrementally smaller) values for `MAX_COLOR`, I'd rather pass that as it's own command-line argument I can keep changing on the fly, instead of repeatedly editing the data file. I can pass data to minizinc using both a dzn data file and separately from the command line by using both the `-d` (for dzn file) and `-D` (for data passed via CLI) flags. Also, I'm going to start kicking off these runs with the `-s` flag for "statistics."

```bash
(colorenv) ~/.../$ minizinc graph_coloring3.mzn -d $prob2.dzn \
                  -D "MAX_COLOR=50;" -p 8 -s --solver org.ortools.ortools
```
(The `-s` is for "statistics", and will give me a bit of statistics on timing (and other things)). Output: 

```bash
% Generated FlatZinc statistics:
%%%mzn-stat: paths=0
%%%mzn-stat: flatBoolVars=13881
%%%mzn-stat: flatIntVars=71
%%%mzn-stat: flatBoolConstraints=13881
%%%mzn-stat: flatIntConstraints=8609
%%%mzn-stat: evaluatedReifiedConstraints=7000
%%%mzn-stat: evaluatedHalfReifiedConstraints=3381
%%%mzn-stat: method="satisfy"
%%%mzn-stat: flatTime=0.525489
%%%mzn-stat-end

46
[1, 2, 3, 4, 5, 6, 7, 3, 8, 9, 10, 10, 1, 11, 12, 13, 6, 5, 2, 4, 5, 7, 9, 13, 11, 14, 12, 15, 16, 17, 18, 19, 20, 19, 20, 21, 21, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 33, 34, 35, 35, 36, 37, 38, 38, 39, 40, 41, 41, 42, 43, 44, 44, 45, 45, 43, 46]
----------
%%%mzn-stat: objective=0
%%%mzn-stat: objectiveBound=0
%%%mzn-stat: boolVariables=9195
%%%mzn-stat: failures=14
%%%mzn-stat: propagations=3929529
%%%mzn-stat: solveTime=2.8972
%%%mzn-stat: nSolutions=1
```
Ok, lots of interesting info once we start using `-s`. But the high-level takeawya here is that running our `solve satisfy max(color) <= MAX_COLOR` version w/ `MAX_COLOR=50` & OR-Tools backend gave us a solution using 46 colors in 2.9 seconds.

Let's try that with the default solver (gecode) as well, by running the same shell command but without `--solver org.ortools.ortools`:

```bash
(colorenv) ~/.../$ minizinc gc3.mzn -d $prob2.dzn -D "MAX_COLOR=50;" -p 8 -s

% Generated FlatZinc statistics:
%%%mzn-stat: paths=0
%%%mzn-stat: flatIntVars=71
%%%mzn-stat: flatIntConstraints=1748
%%%mzn-stat: method="satisfy"
%%%mzn-stat: flatTime=0.0732236
%%%mzn-stat-end

23
[1, 2, 1, 3, 3, 4, 4, 5, 6, 5, 6, 2, 7, 8, 8, 8, 6, 7, 3, 9, 10, 11, 8, 12, 10, 13, 12, 9, 10, 1, 13, 3, 11, 5, 11, 14, 2, 14, 12, 15, 4, 16, 13, 4, 16, 17, 9, 2, 15, 15, 17, 1, 18, 18, 16, 7, 10, 19, 6, 14, 20, 20, 21, 22, 19, 19, 21, 22, 11, 23]
----------
%%%mzn-stat: initTime=0.004835
%%%mzn-stat: solveTime=0.002239
%%%mzn-stat: solutions=1
%%%mzn-stat: variables=71
%%%mzn-stat: propagators=1631
%%%mzn-stat: propagations=5583
%%%mzn-stat: nodes=66
%%%mzn-stat: failures=0
%%%mzn-stat: restarts=0
%%%mzn-stat: peakDepth=55
%%%mzn-stat-end
%%%mzn-stat: nSolutions=1

```
Wow. We got a much better solution (23 colors instead of 46), and it takes about 0.002 seconds, wayyy faster than OR-Tools. So much for that Minizinc Challenge gold medal? Also interesting to compare how the compilation stats ("Generated FlatZinc statistics") and solver/runtime stats (those below the solution) vary between OR-Tools and gecode. But will have to defer that discussion for now.

So since gecode found a solution with only 23 colors, let's try and see if we can find 20 (again w/ default solver gecode)...
```bash
(colorenv) ~/.../$ minizinc gc3.mzn -d $prob2.dzn -D "MAX_COLOR=50;" -p 8 -s
```
Found immediately (0.2 seconds). OR-tools?
```bash
(colorenv) ~/.../$ minizinc gc3.mzn -d $prob2.dzn -D "MAX_COLOR=50;" -p 8 -s --solver org.ortools.ortools
```
Also solved...in 0.7 seconds! OR-Tools got faster when we gave it a much more stringent problem. Wild.

- 19 colors? Sub-second solution for both gecode and OR-tools. 
- 18? ditto
- 17? took 25 seconds for gecode, OR-tools hangs for at least a couple minutes, then I killed it. Tried another built-in Minizinc solver called "chuffed," also hung for at least a couple minutes. Also noteworthy - chuffed (per my sys-monitor) only used 1 core, even with `-p 8`.
- 16 for MAX_COLOR hangs for at least a couple minutes on each solver.

Alright, so let's submit that presumably-not-optimal 17-color solution we got with gecode for Problem2 to the coursera grader with `submit.py` and see what we score...

10/10! 

So does that mean 17 is optimal? Not sure. The best solution I can find a record of anyone getting on the forums is 17. But that doesn't mean it's optimal. It does, along with the 10/10 grade, probably mean that it's at least pretty good. 

### Problems 3-6 Using the Solve Satisfy Strategy
Ok so what did we just learn? 17 colors on problem 2 is apparently a very good solution, good enough for 10/10 points, but:
 - We do not know if it's optimal
 - It would take an indeterminate amount of compute power (or some Minizinc modeling improvement I'm not aware of, or some completely different approach) to prove (glboal) optimality)

Let's proceed with this (albeit manual) iterative process of ratcheting down the `MAX_COLOR` we `solve satisfy` for, for the rest of our problems (3-6) and see how those do...

#### Problem 3
```bash
(colorenv) ~/.../$ head -1 $prob3 
100 2502
```
So 100 Nodes, 2,502 Edges. Roughly 30-40% larger than those numbers for problem2, although that does not imply linear growth in solve time. Jumping in w/ MAX_COLOR=50:

```
(colorenv) ~/.../$ minizinc gc3.mzn -p 8 -s -D  'MAX_COLOR=50;' -d $prob3.dzn
```
...solves immediately.
- with 40? same
- 30? yup
- 20? uh huh 
- 19? 18? yawn
- 17? now this took 7.1 seconds. 
  - Interesting because that's as good as we could get on problem2, which took 25 seconds and that had fewer nodes/edges.
  - Also intersting... ortools w/ `-p 8` (8 threads (on 8-core machine)) solves in 2.1 seconds. Significantly faster than gecode on problem3, after performing significantly worse on $prob2.
    - Also also intersting: OR-tools solves faster (1.5seconds) with `-p 4` & `-p 2` (re-run multiple times to confirm faster than `-p 8`, and then hangs (I killed it after two minutes) for a single-threaded run (`-p 1` or without `p` option specified). This is all very confusing. 
  - And chuffed solves for MAX_COLOR=17 in 0.6seconds! Another als-ran on problem2.
- 16?
  - gecode (w/ `-p 8`) hangs (I killed it after 2 minutes)
  - OR-tools (w/ `-p 8`) solves in 32.9 seconds
  - Chuffed hangs (killed after 2 minutes)

Submitting that MAX_COLOR=16 solve-satisfy solution for problem3 w/ OR-tools?
10/10. Sweet. 

Although there is a record of some people on the forums getting a 15-coloring. So 16 is definitely not optimal. Could our model have found the 15-coloring with a bit more time? I'll kick it off and go get a bagel to see... 

Ok after 25 minutes, apparently no.

#### Problems 4
```bash
(colorenv) ~/.../$ head -1 $prob4
250 28046
```

MAX_COLOR = 95
- 50.9s for ortools
- 0.1s for gecode
- 1.6s for chuffed

MAX_COLOR = 94
- 52.8s for ortools
- 13.0s for gecode
- 2.7s for chuffed

MAX_COLOR = 93
- 61.5s for ortools 
- 172.6s (2min 52.6s) for gecode
- 2.9s for chuffed
One last interesting thing on this run: this was the first run I've seen where a significant amount of time was spent on compiling the Minizinc code down to Flatzinc. OR-tools and Chuffed each spent >5seconds just on that step. Gecode, by contrast, compiled the problem down to Flatzinc in 0.4seconds, but then took by far the longest to solve it (for MAX_COLOR=93).

You may be wondering - why would the choice of backend solver affect the compile-time down to Flatzinc? I said up top that Minizinc compiles your Minizinc code down to Flatzinc, and then the backend solvers solve the Flatzinc problems. It turns out, probably because most of these Flatzinc solvers are mostly used via the Minizinc language, that they can each specify their own implementations for compiling certain Minizinc functions down to Flatzinc. You can also affect the compilation itself by passing optional minizinc CLI params of `-O2`, `-O3`, `-O4` or `-O5`.


MAX_COLOR = 92
- each of OR-tools, gecode and chuffed all hung - I interrupted each after 3-5 minutes.

Finally, submitted the 93-coloring...7/10.

#### Problems 5
```bash
(colorenv) ~/.../$ head -1 $prob5
500 12565
```
So twice as many nodes and a little less than half as many edges as Problem4.

Gecode really dominates this one. Solves everything from MAX_COLOR=20 all the way down to 15 in under 1second (compile time + solve-time). By MAX_COLOR=14, after four minutes of not solving I interrupted it.

Chuffed and OR-tools not as good here. Submitting that gecode 15-coloring...
10/10

#### Problems 6
```bash
(colorenv) ~/.../$ head -1 $prob6
1000 249482
```
Oh wow. 1,000 Nodes, 249,482 edges. Yeesh.

For this problem, gecode again did consistently better than chuffed and OR-tools. For gecode I got down to `MAX_COLOR=122`. Then trying 121, it hung. Then I tried using the `-O2` optimization flag, which supposedly performs a second compilation pass, but that exploded memory requirements and ate all my RAM + swap. I killed that, backtracked and tried `MAX_COLOR=123` with another supposedly optimized compilation flag, `-O3`, but that caused both compilation *and* (curiously) solve time to take longer than before. Submitted the 122-coloring solution... 7/10. 

### Wow that was manual
So this solve-satisfy strategy, iterating on progressively lower `MAX-COLOR` values actually worked pretty well. We ultimately arrived at four solutions which scored 10/10, and two which scored 7/10, for a total of 54/60. All this with a remarkably clear and concise declarative bit of minizinc code:
```minizinc
include "globals.mzn";

% Input Parameters (values for these specified in .dzn data file)
int: NUM_NODES; 
int: NUM_EDGES;
int: MAX_COLOR;
array[1..NUM_EDGES,1..2] of int: edges;

% Decision Variable: the solver will find the values of these
array[0..NUM_NODES-1] of var 1..MAX_COLOR: color;

% Problem-Statement Constraint
constraint forall(e in 1..NUM_EDGES)(color[edges[e,1]] != color[edges[e,2]]);

% Symmetry-Breaking Constraint
constraint forall(n in 1..NUM_NODES-1)(value_precede(n, n+1, color));

% Replacing Objective Function here with Solve Satisfy to <= CLI-arg MAX-COLOR
constraint max(color) <= MAX_COLOR;
solve satisfy;

% formatted output
output[show(color)];
```

The downside is, at least how I did it here in my first real test drive with Minizinc, it took an awful lot of manual fiddling; trying out different solvers, `MAX_COLOR` values and occasionally other config parameters for each problem. And frustratingly, finding that one solver was faster than the others for a given problem for at e.g. `MAX_COLOR=n` did not at all mean that it would also be the fastest or most likely to solve for `MAX_COLOR=n-1` or `n-2`, etc. 

What I'd really like is a way to kick off a script where you get an ok solution quickly, better solutions over time, and you can kinda let it run arbitrarily long if you want to keep waiting on the machine to search for better and better solutions. Originally I hoped I could use Minizinc's `--all-solutions` CLI flag for this purpose. But it didn't quite work out. Here's why:

Recall for problem2, we got our best solution of 17-colors after 25 seconds with gecode. We can find a `MAX_COLORS=18` solution in under a second. What happens if we try with say `MAX_COLORS=25` and let `--all-solutions` run? (note below I adjusted the output to just print out the `max(color)` of each solution, instead of the full sequence). After 2 minutes, here's what I have:

```bash
(colorenv) maxepstein@pop-os:~/discrete_opt/graph coloring week3$ minizinc gc4.mzn -d "$prob2.dzn" -p 8 -D "MAX_COLOR=25;" --all-solutions

23
----------
24
----------
25
----------
22
----------
```
So we get a bunch of solutions around our `MAX_COLOR`, but not necessarily ones getting lower. Which makes sense. Even after much more time than it took to solve for 17 or 18 directly. 

I then thought using using [search-annotations](https://www.minizinc.org/doc-2.3.1/en/mzn_search.html) with restarts would help, but I was unable to get that to work either. The documentation suggested using search params that would lead to randomized search, so that restarts didn't lead the solver deterministically right back to where it was before the restart. This makes a lot of sense, but when I tried the randomizing search params `indomain_random` and `dom_w_deg`, I got messages that they are each not implemented for the solver. But I should try and nail down exactly which solvers they are implemented for. 

To the extent I want a solution that sorta automates the guess-and-check solve-satisfy approach iterating through more and more ambitious `MAX_COLOR` values, like I did manually with this assignment, I probably want to script that myself in Python. Which is fine. In my experience it's generally good to go through the exercise of playing around with a bunch of different options and alternatives manually first, before trying to automate it. But it would also be good for me next time to compare the effect of various solvers/config settings for a given problem (or several) in a scripted and so more comprehensive way.

### Final Takeaways
Playing around with Minizinc has been really educational and pretty fun. I learned a fair amount about minizinc modeling, symmetry-breaking, different config options and backend solvers, although I clearly have much more to learn in each of those areas. However, what accelerated my progress the most on this assignment at least was actually just the change in perspective on how to use Minizinc. Specifically, going from 
- "Minizinc will find me a (provably) optimal solution, how can I make it run faster?" to
- "Minizinc can give me a feasible solution quickly. Let's iterate a bit and see how good of a solution we can get."

My experience here was that I could actually get a very good solution very quickly, at least for one of the backend solvers, and at least with a bit of symmetry-breaking, for even very large problems.

So what would I do going forward to bump it up to 60/60? There's some additional things I'd like to try out in Minizinc, but honestly for a hard optimization problem I think I'd rather use Minizinc as 'part of a balanced breakfast' than as the whole meal. I would use the 7/10 feasible solution I already got (or maybe several different 7/10 feasible solutions I can get quickly from Minizinc) to seed a local-search algorithm to make additional improvements.

But I also really want to try out a bunch of different things in Minizinc going forward. In particular, I'd like to grid-search a bunch of backend-solver options, compilation options, and [search annotation](https://www.minizinc.org/doc-2.3.1/en/mzn_search.html) settings for a given problem instance to get a more comprehensive picture of how Minizinc performance varies across all these parameters. I also need to re-read [this part of the documentation](https://www.minizinc.org/doc-2.3.1/en/efficient.html) on effective modeling practices in Minizinc. Clearly how you code the model itself has a huge effect on performance, as we already saw with symmetry-breaking.