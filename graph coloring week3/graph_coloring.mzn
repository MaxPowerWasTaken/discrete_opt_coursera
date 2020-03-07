% Input Parameters (values for these specified in .dzn data file)
int: NUM_NODES; 
int: MAX_COLORS = NUM_NODES;
int: NUM_EDGES;
array[1..NUM_EDGES,1..2] of int: edges;

% Decision Variable: the solver will find the values of these
array[0..NUM_NODES-1] of var 1..MAX_COLORS: color;

% Our Constraints 
constraint forall(e in 1..NUM_EDGES)(color[edges[e,1]] != color[edges[e,2]]);

% Our Objective Function
solve minimize max(color);

% formatted output
output[show(color)];