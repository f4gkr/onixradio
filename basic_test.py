import numpy as np
import sclblpy as sp  # Import the sclblpy package
import sclblonnx as so

g = so.empty_graph()

# Add a node to the graph.
n1 = so.node('Add', inputs=['x1', 'x2'], outputs=['sum'])
g = so.add_node(g, n1) 

# Add inputs:
g = so.add_input(g, 'x1', "FLOAT", [1])
g = so.add_input(g, 'x2', "FLOAT", [1]) 

# And, add an output.
g = so.add_output(g, 'sum', "FLOAT", [1])

# First, let's clean the graph (not really necessary here) 
g = so.clean(g) 
# Next, lets see if it passes all checks:
so.check(g)
so.graph_to_file(g, "basic.onnx")
 

# Evaluate the graph:
example = {
    "x1": np.array([1.2]).astype(np.float32), 
    "x2": np.array([2.5]).astype(np.float32)
   }

result = so.run(g,
    inputs=example,                
    outputs=["sum"])

print(result)