# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:15:12 2021

@author: burak
"""
import numpy as np
import jax
import tensornetwork as tn

# Next, we add the nodes containing our vectors.
a = tn.Node(np.ones(10))
# Either tensorflow tensors or numpy arrays are fine.
b = tn.Node(np.ones(10))
# We "connect" these two nodes by their "0th" edges.
# This line is equal to doing `tn.connect(a[0], b[0])
# but doing it this way is much shorter.
edge = a[0] ^ b[0]
# Finally, we contract the edge, giving us our new node with a tensor
# equal to the inner product of the two earlier vectors
c = tn.contract(edge)
# You can access the underlying tensor of the node via `node.tensor`.
# To convert a Eager mode tensorflow tensor into 
print(c.tensor)


a = tn.Node(np.eye(2))
# Notice that a[0] is actually an "Edge" type.
print("The type of a[0] is:", type(a[0]))
# This is a dangling edge, so this method will 
print("Is b[0] dangling?:", b[0].is_dangling())

trace_edge = a[0] ^ a[1]
# Notice now that a[0] and a[1] are actually the same edge.
print("Are a[0] and a[1] the same edge?:", a[0] is a[1])
print("Is a[0] dangling?:", a[0].is_dangling())

'''
Axis naming.
Sometimes, using the axis number is very inconvient and it can be hard to 
keep track of the purpose of certain edges. To make it easier, 
you can optionally add a name to each of the axes of your node. 
Then you can get the respective edge by indexing using the name instead of the number.'''


''' 
Section 2. Advanced Network Contractions
Avoid trace edges.
While the TensorNetwork library fully supports trace edges, 
ontraction time is ALWAYS faster if you avoid creating them. 
This is because trace edges only sum the diagonal of the underlying matrix, 
and the rest of the values (which is a majorit of the total values) are just garbage. 
You both waste compute time and memory by having these useless trace edges.

The main way we support avoid trace edges is via the @ symbol, which is an alias to 
tn.contract_between. Take a look at the speedups!


'''
# Here, a[0] is a['alpha'] and a[1] is a['beta']
a = tn.Node(np.eye(2), axis_names=['alpha', 'beta'])
edge = a['alpha'] ^ a['beta']
result = tn.contract(edge)
print(result.tensor)

def one_edge_at_a_time(a, b):
  node1 = tn.Node(a)
  node2 = tn.Node(b)
  edge1 = node1[0] ^ node2[0]
  edge2 = node1[1] ^ node2[1]
  tn.contract(edge1)
  result = tn.contract(edge2)
  return result.tensor

def use_contract_between(a, b):
  node1 = tn.Node(a)
  node2 = tn.Node(b)
  node1[0] ^ node2[0]
  node1[1] ^ node2[1]
  # This is the same as 
  # tn.contract_between(node1, node2)
  result = node1 @ node2
  return result.tensor

a = np.ones((1000, 1000))
b = np.ones((1000, 1000))
print("Running one_edge_at_a_time")
%timeit one_edge_at_a_time(a, b)
print("Running use_cotract_between")
%timeit use_contract_between(a, b)


''' 
We also have `contract_parallel` which does the same thing as
`contract_between`, only you pass a single edge instead of two nodes.
 This will contract all of the edges "parallel" to the given edge
 (meaning all of the edges that share the same two nodes as the given edge)

Using either method is fine and they will do the exact same thing. 
In fact, if you look at the source code, `contract_parallel` actually just calls `contract_between`. 
:)

'''


def use_contract_parallel(a, b):
  node1 = tn.Node(a)
  node2 = tn.Node(b)
  edge = node1[0] ^ node2[0]
  node1[1] ^ node2[1]
  result = tn.contract_parallel(edge)
  # You can use `get_final_node` to make sure your network 
  # is fully contracted.
  return result.tensor

print("Running contract_parallel")
%timeit use_contract_parallel(a, b)

'''

Complex Contraction.
Remember this crazy hard to write tensor contraction? Well, we're gonna do it in about 13 lines of simple code.
'''

# Here, we will contract the following shaped network.
# O - O
# | X |
# O - O
a = tn.Node(np.ones((2, 2, 2)))
b = tn.Node(np.ones((2, 2, 2)))
c = tn.Node(np.ones((2, 2, 2)))
d = tn.Node(np.ones((2, 2, 2)))
# Make the network fully connected.
a[0] ^ b[0]
a[1] ^ c[1]
a[2] ^ d[2]
b[1] ^ d[1]
b[2] ^ c[2]
c[0] ^ d[0]
# We are using the "greedy" contraction algorithm.
# Other algorithms we support include "optimal" and "branch".

# Finding the optimial contraction order in the general case is NP-Hard,
# so there is no single algorithm that will work for every tensor network.
# However, there are certain kinds of networks that have nice properties that
# we can expliot to making finding a good contraction order easier.
# These types of contraction algorithms are in developement, and we welcome 
# PRs!

# `tn.reachable` will do a BFS to get all of the nodes reachable from a given
# node or set of nodes.
# nodes = {a, b, c, d}
nodes = tn.reachable(a)
result = tn.contractors.greedy(nodes)
print(result.tensor)


# To make connecting a network a little less verbose, we have included
# the NCon API aswell.

# This example is the same as above.
ones = np.ones((2, 2, 2))
tn.ncon([ones, ones, ones, ones], 
        [[1, 2, 4], 
         [1, 3, 5], 
         [2, 3, 6],
         [4, 5, 6]])



# To specify dangling edges, simply use a negative number on that index.

ones = np.ones((2, 2))
tn.ncon([ones, ones], [[-1, 1], [1, -2]])


# To make the singular values very apparent, we will just take the SVD of a
# diagonal matrix.
diagonal_array = np.array([[2.0, 0.0, 0.0],
                           [0.0, 2.5, 0.0],
                           [0.0, 0.0, 1.5]]) 


# First, we will go over the simple split_node method.
a = tn.Node(diagonal_array)
u, vh, _ = tn.split_node(
    a, left_edges=[a[0]], right_edges=[a[1]])
print("U node")
print(u.tensor)
print()
print("V* node")
print(vh.tensor)


# Now, we can contract u and vh to get back our original tensor!
print("Contraction of U and V*:")
print((u @ vh).tensor)

'''
We can also drop the lowest singular values in 2 ways,

By setting max_singular_values. This is the maximum number of the original singular values that we want to keep.
By setting max_trun_error. This is the maximum amount the sum of the removed singular values can be.

'''
# We can also drop the lowest singular values in 2 ways, 
# 1. By setting max_singular_values. This is the maximum number of the original
# singular values that we want to keep.
a = tn.Node(diagonal_array)
u, vh, truncation_error = tn.split_node(
    a, left_edges=[a[0]], right_edges=[a[1]], max_singular_values=2)
# Notice how the two largest singular values (2.0 and 2.5) remain
# but the smallest singular value (1.5) is removed.
print((u @ vh).tensor)

# truncation_error is just a normal tensorflow tensor.
print(truncation_error)