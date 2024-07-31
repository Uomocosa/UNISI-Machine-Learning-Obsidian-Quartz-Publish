# Regression and Classification
**Linear Regression** : $f_{w}(x) = b + w_1x_1 + w_2x_2 + \ldots$
**Classification** : $f_{w}(x) = \sigma(b + w_1x_1 + w_2x_2 + \ldots)$

**Error function**:
$$
E = \sum_{k=1}^l V\left(f\left(x_k,w_k\right), y_k\right)
$$
**Loss function**: 
$$
V\left(f\left(x_k,w_k\right), y_k\right)
$$ 
The loss function defines how distant or different is the evaluation of one single set of features from the actual solution.

**inputs**: $x$
**weights**: $w$
**real outputs**: $y$
**estimated outputs**: $f(x_k,w_k)$

---
# Gradient Descent
Is an **algorithm**
It works with the **linear** regression formula.
Works well when given a lot of parameters and runs really fast.

---
# Empirical Derivative
To calculate a derivative if we don't care too much about velocity we can use its definition:
$$
\frac{\partial f}{\partial t} = \lim_{\partial t \to 0} \frac{f(t+\partial t) - f(t)}{\partial t}
$$
And using a $\delta t$ really small evaluate it empirically

---
# Stochastic Gradient Descent
Instead of updating the gradient after calculating all the error function, we update it after calculating only the Loss function (1 step of gradient descent for every example taken)

There is a good change that the stochastic gradient descent actually brings us to the local minimum

Differences with **Normal Gradient Descent**: we have the security that it will brings us to the local minimum

---
# Batch Gradient Descent
The middle ground, i update the gradient (1 step) after looking at an arbitrary number of Loss functions

Differences with **Normal Gradient Descent**: I update the gradient (1 step) after looking at the sum of **all** Loss functions (equal to the Error function)

The Normal gradient descent is the most mathematically correct one, but it's not always the best choice, sometimes is not even doable:
> ~Ex.: When gathering data from an online website for example, the data stream is infinite, we cannot use the normal gradient descent but only the stochastic or batch one

---
# Forgetting behaviour
With stochastic gradient descent and batch gradient descent you learn only the last samples and tend to forget the previous ones.

---
# Calculus Chain Rule
Remember that given the loss function:
$$
e_k := V\left(f\left(\hat{w}, \hat{x}_k\right), y_k\right)
$$
Its derivative can be rewritten as:
$$
\frac{\partial}{\partial w_i}e_k = \frac{\partial V}{\partial f}\frac{\partial f}{\partial w_i}
$$
---
