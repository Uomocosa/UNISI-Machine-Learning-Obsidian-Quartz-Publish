# Normal Equations
- Linear prediction in multi-dimensional spaces
- Normal equations and projections
- Different cases and pseudo-inversion

---
## Polynomial Approximation:
In case of a polynomial function: 
$$
f = w_2x^2 + w_1x b
$$
we can see it as a liner multi-variable function:
$$
f = w_2x_2 + w_1x_1 + b
$$

---
# Normal Equation - Solutions
To solve a normal equation we will need **more** points than unknown variables, which means having enough data to learn the weights.

Given the matrix $\hat{X}$ : "example" matrix, or data matrix, we put in each **row** one observation, plus the bias input (usually 1).
The number of **columns** correspond to the total number of observation we took.

- More rows = More **parameters** = More **weights** to learn but a more "complete" solution, i.e. where the edge cases of the problem are taken into account
- More columns = More **data** = More precise **weights**

> Number of **columns** $\gg$ Number of **rows**

Given:
- $\hat{X}$ : **information** matrix
- $\hat{W}^*$ : "perfect" **weight** matrix: such that $(\hat{X}^T\kern 1px\hat{X})\kern 1px\hat{W}^* = \hat{X}\kern 3px Y$
- $Y$ : **target** Matrix

We have that: $\operatorname{rank}(\hat{X}) = r \le d$ where $d$ is the maximum rank
And we can say that:
$$
(\hat{X}^T\kern 1px\hat{X})\kern 1px\hat{W}^* = \hat{X}\kern 3px Y
$$
So:
$$
\hat{W}^* = (\hat{X}^T \kern 1px \hat{X})^{-1} \kern 1px \hat{X}^T \kern 3px Y
$$
---
