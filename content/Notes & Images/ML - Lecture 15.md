# Definition: Feedforward NN
FNN is an architecture which is defined by a **DAG** (Directed Acyclic Graph)

**DAG**:
$G \sim (V, A)$
meaning: the graph contains the vertices and arches.

where: $V = \{v_1, \ v_2, \ \ldots, v_n \}$ 
(vertices).

and, $A = \{ a_1, \ a_2, \ \ldots, \ a_m \}$ 
(arches).

with: $a_k \sim (v_i, v_j)$ 
(arch from $v_i$ to $v_j$).

$a_k \sim (v_i, v_j)$  is permitted only if $i \prec j$ ($i$ precedes $j$) in the relationships.

---
# Backpropagation
The Backpropagation algorithm works only for FNN.
Even if they have more than 1 hidden layer (MLN - Multi Layered Neural Networks)

> MLN are FNN

