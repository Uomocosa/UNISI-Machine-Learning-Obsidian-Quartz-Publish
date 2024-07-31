# DAG Partial Ordering
![[Pasted image 20220207182210.png]]![[Pasted image 20220207182218.png]]

Both of this figures represent DAGraphs the second is defined as partial ordered because it can be clearly divided into 4 layers.

We often use the DAG partially ordered for NN because when programmed we can see each layer as a **matrix** which can be modelled really efficiently.

But it's important to notice that in some cases having some **"skips"** (which mean for example to create an arch from the 1st and 4rd layer), so breaking the partial ordering, can be beneficial.

---
# RNN
**R**ecurrent **N**eural **N**etwork
![[Pasted image 20220207183309.png]]

**ATTENTION**: This topology of NN is not a DAG

To compute the calculation according to this graph we will need to introduce the concept of time.

Luckily we can use the procedural way in which machine works (one instruction at a time to) make a concept of time.

Also we can define if needed a max number of loops done by a RNN, or we cannot and obtain a continuous, infinite output.

---
###### Backpropagation Algorithm
[BRILLIANT](https://brilliant.org/wiki/backpropagation/#)
Look it up on Coursera or YouTube.
