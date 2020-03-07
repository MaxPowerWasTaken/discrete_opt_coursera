## Constraint Programming & The Graph Coloring Problem 
Week 3 of [Discrete Optimization](https://www.coursera.org/learn/discrete-optimization/home/week/3) on Coursera covered Constraint Programming, and the programming assignment is the Graph Coloring Problem.

Constraint Programming(CP) seems to me to be a very elegant approach. You start by describing the structure of a problem, by coding:
- parameters, 
- decision variables, 
- constraints, and 
- an objective function. 

This problem description can be translated to, or implies, a certain space of possible solutions.

The CP language will first use the constraints given to eliminate as many theoretical solutions as possible; i.e. to prune the solution space. It will then proceed with a possible value for one of the decision variables, re-check the constraints to again further prune the solution space, rinse and repeat. Often it will encounter that the set of decision variable values it has assigned so far has led it to an infeasibility, in which case it will backtrack and change a previously assigned decision variable, again prune the search space, and proceed again.

The idea is to attack an NP hard problem with a large space of possible solutions by aggressively ruling out as much of the theoretical solution space as possible. Because we need constraints to prune the solution space, adding more constraints to our problem description can be really helpful, even if, to a mere human, they may seem redundant.

I figured this would be a good time to try out the Constraint Programming language [Minizinc](https://www.minizinc.org/) some more, which I mentioned briefly at the end of my [knapsack blog post](https://maxpowerwastaken.gitlab.io/model-idiot/posts/knapsack_blog/). 

### The Graph Coloring Problem 
We are given an undirected graph; a set of nodes and edges where each edge connects two nodes. Each node has an attribute: a color. The problem is to find the set of colors for the nodes that:
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

### Diving into some Minizinc code 
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

We can run this in the Minizinc IDE or from shell:
```bash
maxepstein@pop-os:~/discrete_opt/graph coloring week3$ minizinc graph_coloring.mzn
color = array1d(0..3, [2, 1, 2, 2]);
----------
==========
```
Ok so the model finds the optimal solution, using two colors. Color '2' for the first, third and fourth node, and color '1' for the second.

#### First Reactions to our MiniZinc Code  
This to me is wild. Really super cool. Yeah this was a very small dataset, and yeah the syntax for Minizinc probably looks a little unfamiliar. But all I did was write a description of the problem - none of the code above really deals with *how* to solve the problem at all. And we get a solution which is proved optimal. 

So that's it? We describe the problem and we're just done?
Well not really. Like before, as we scale the size of the dataset (based on my limited experience with Minizinc) we'll still run into some issues. And then we refine how we formulated the problem, or try out different backend solvers (Minizinc is compatible with many), or other tricks to help. But anyway, I'm excited to see how far Minizinc will take us here.

### Extending Our Minizinc Solution
In the Minizinc code above, I added the data from our data file to the Minizinc code, and formatted it to be proper Minizinc syntax. Moving forward, to run this model on arbitrary data files, we need to take two steps:
- Separate the modeling code from the data-input code (that assigns values to parameters)
    - This can be done by moving the 'data input' code into a `.dzn` minizinc data file.
- Write a script that takes the data input files we're given, and reformats them to suitable `.dzn` minizinc data files

#### Separating Model From Data in Minizinc
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

So here's our first `solver.py`, which basically:
- reformats the input datafile to a .dzn minizinc data file
- calls our minizinc model
- reformats the output per the coursera course requirement

`solver.py`:
```python
import os

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



def solve_it(input_data):
    minizinc_model = "minizinc graph_coloring.mzn"

    # Generate .dzn file for minizinc model
    convert_data_to_dzn(input_data, dzn_filename = 'data.dzn')

    # Run Minizinc Model
    stream = os.popen(f'{minizinc_model} mzn_data/data.dzn')
    mz_output = stream.read()

    # Parse Minizinc Output and Prepare Coursera Course Output
    OPTIMAL = 0
    if "==========" in mz_output:
        OPTIMAL = 1

    colors = mz_output.split('\n')[0].strip('[]').split(', ')
    num_colors = len(set(colors))

    # prepare the solution in the specified output format
    output_data = f"{num_colors} {OPTIMAL}\n"
    output_data += ' '.join(colors)

    return output_data
```
And to confirm 


#### errata
The `----------` in the output below `color...` means MiniZinc found a solution, and the `==========` means MiniZinc proved that solution is optimal.