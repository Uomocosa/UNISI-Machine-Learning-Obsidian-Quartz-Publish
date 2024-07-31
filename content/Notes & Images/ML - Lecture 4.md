# Linear prediction
- Best fitting principle
- Least Mean Square (LMS)
- What if the function is not linear?

---
# Terminology
- $x$ : input
- $w$ : weight
- $b$ : bias
- $E$ : Error or Cost function
- $L$ : Loss function

---
# Linear Neuron
$$
f(x) = wx + b
$$

![[Pasted image 20220205201335.png]]

Error or Cost function:
$$
E(w,b) = \sum_{k = 1}^{l}(y_k - wx_k - b)^2
$$
Given the **error function** it's easy to find the best **weights** to solve the problem.
We just have to find $\vec{w} = \{w_1, \ w_2, \ \ldots\}$ and $b$ such that the cost is minimized, which are given by:
$$
\nabla E(w,b) = 0
$$
Which is equal of saying:
$$
\left\{
	\begin{align}
	\frac{\partial E}{\partial w} = 0
	\\[5px]
	\frac{\partial E}{\partial b} = 0
	\end{align}
\right.
$$

^7703b6

This system often doesn't accept a solution, but we can be satisfied by approximating the solution.

---
# Variance
Given a random variable $x$ and its average ($\bar{x}$) the variance of $x$ is defined as:
$$
\hat{\sigma}_{xx}^2 = \frac{1}{l} \sum_{k = 1}^l (x_k - \bar{x}_k)^2
$$
The variance is a measure on **data sparsity**
If the variance is really high it tells us that the variable assume 
a many different values.
While if it's small, then the variable as almost always the same value

---
# Cross-Correlation
Given two random variables $x$ and $y$ and their mean values ($\bar{x}, \bar{y}$): their cross correlation is defined as:
$$
\hat{\sigma}_{xy}^2 = \frac{1}{l} \sum_{k = 1}^l (x_k - \bar{x}_k)(y_k - \bar{y}_k)
$$

When the cross correlation is really small, we can say that between the two random variable there is no **dependency**

---
### What if the problem is Non Linear?
Let's take for example the following problem 'space needed to stop a car' :
$$
s = \frac{v^2}{2a}
$$
To calculate **variance** and **cross correlation** we can assume:
$$
\begin{align}
x = v^2
\\
y = 2a
\end{align}
$$