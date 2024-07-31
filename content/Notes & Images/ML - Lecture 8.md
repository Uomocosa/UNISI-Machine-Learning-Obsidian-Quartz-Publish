# Ridge Regression
- The power of regularization
- Ridge regression idea
- Statistical notes

The Ridge Classifier, based on Ridge regression method, converts the label **data** into $[-1, 1]$ and solves the problem with regression method. The highest value in prediction is accepted as a target class and for multiclass data multi-output regression is applied.

---
# Neurons
![[Pasted image 20220206164011.png]]

Where sigma can be **any** function, typically tho it will return outputs between $[0, \ 1]$ 
The most common **activation function** ($\sigma$) is the **sigmoid function**:
![[Pasted image 20220206164230.png]]

---
# Saturation
The sigmoid function, like many other activation functions, can be subjected to a phenomenon called **saturation** of the neuron.

This happens when the weight becomes too large (positive or negative) and no matter of the inputs $\vec{x}$ the resulting output: $\sigma(w_1x_1 + w_2x_2 + \ldots + b)$ will always be 0 or 1 nothing in between.

The algorithm do solve this problem automatically but it could take a lot of time, so its best to adopt some strategies to stop the saturation from happening.

---
# Classification: 
Suppose we want to separate 2 sets of points, black from gray, we define **Linear Separability** as a property of the set when is possible to draw a **straight** line that completely separates the two sets.

**Definition**:
Given a collection of points $L = \{(x_k, y_k)\} \  \text{for} \ k = 1, l$ 
where $x_k \in \mathbb{R}^{d+1}$ are the parameters and $y = \{0  \ \text{or} \ 1\}$ is the class
We say that $L$ is **linearly separable** if there exist $\hat{w}$ such that:
$$
\sigma\left(\hat{w}^T \kern 1px \vec{x}\right) = \left\{
	\begin{align}
	&1 \ \text{if $k$ is positive}
	\\
	&0 \ \text{if $k$ is negative}
	\end{align}
\right.
$$

![[Pasted image 20220206165638.png]]

---
### Linear Separability in the Boolean function:
![[Pasted image 20220206171527.png]]

The _XOR_ function is not linearly separable

---
# Reference to probability
**Remember**:
When measuring a random variable the distribution of the observation can be expected to take a Gaussian Distribution.

- [[DES - Law of Large Numbers|Law of Large Numbers]]
- [[DES - Central Limit Theorem|Central Limit Theorem]]

Given this knowledge then we can say that for a problem like "Determine if a persone is male or female given their height"
Then plotting the data we gather we can expect a graph like this:
![[Pasted image 20220206172539.png]]
The purple line represent the distribution of female heights
The red line the male heights
Then the point I'm looking for is exactly:
![[Pasted image 20220206172645.png]]
Because the distribution are the same, we can say that the point is found equaling:
$$
P(♀ \mid X = x) = P(♂ \mid X = x)
$$
From this formula we find the intersection point.