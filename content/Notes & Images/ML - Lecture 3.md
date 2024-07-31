# Adversarial ML
Perturbing an image before giving it to the NN can alter Massively the results
![[Pasted image 20220205172457.png]]

---
# Supervised Learning
- Regression
- Classification
- Loss Function

The **supervisor** tells you how wrong you are.
**Learning** means to respect the supervision (determine the right weights such that the input corresponds to the outputs)

**Learning Environment**: Collection of **inputs** and the corresponding **outputs**.

---
# Regression
Given the _height $h$ of a person_ we want to predict his _weight_ as a function oh $h$: $w(h)$, from a collection of supervised data

**We need an output corresponding to a real number $\in \mathbb{R}$**


---
# Classification
Determine if a *picture* represent a number between 0-9.

**We need an output corresponding to a set of numbers, more generally corresponding to a category$\in \mathbb{N}$**

---
### Measure the Learning Process:
In case of supervised learning a measure of the learning process can be represented by the **Loss**: a sort of distance between the machine-found output and the actual output:

![[Pasted image 20220205195829.png]]

Our objective is to minimize: $|\omega - f(h)|$, defined as the **Loss Function**.

There exist infinite loss functions another one can be: $(\omega - f(h))^2$ or $\operatorname{log}|\omega - f(h)|$

In case of classification the loss function becomes a little more complicated because we are working with discrete numbers (or classes), but we can still find a loss function.

---
# Unsupervised Learning
- Linear Separation (data clustering)


Find **patterns** in the data.

---
# Linear Separation
![[Pasted image 20220205200445.png]]

To perform data clustering we can use the geometric distance of two points:
- Given the data point $x$ and two **cluster points** $o_1$ and $o_2$ with define with $d(x, y)$ the distance between 2 points.
- If $d(x_1, o_1) < d(x_1, o_2)$ than $x$ will belong to the cluster $o_1$ else, to the cluster $o_2$ 

---
## Using the right tool for the job
![[Pasted image 20220205201008.png]]

Using the loss function shown in the picture is actually a bad idea.
In this case the machine thinks that the 3 is more similare the the 2 to the reference image.