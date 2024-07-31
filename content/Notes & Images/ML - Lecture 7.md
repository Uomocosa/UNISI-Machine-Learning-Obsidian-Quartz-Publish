# Regularization
If you are given more variables (features) than learning example, you could use the regularization algorithm to reduce linear dependent parameters, reducing their number

> **NOTE**:
> The Regularization tecniche is kind of dated.
> Usually the number of examples is much grater than the number of parameters (also called features or variables), but this is **NOT** true for top level experiments in Deep Neural Networks, where we work with $10^9$ parameters but only with a pool of $10^6$ examples, but for DNN the Regularization tecniche is **NOT** used.

---
### Extension of the Information Matrix
Continuing using the linear regression, we can extend the Information Matrix $\hat{X}$ with some non-linear terms.

We can add a column that consist of all the square elements of the first parameter: $x_1^2$.
Resulting in:
$$
\hat{X} = \left[
\begin{array}{}
&x_{1 \ (1)} & x_{2 \ (1)} & x_{2 \ (1)}
\\
&x_{1 \ (2)} & x_{2 \ (2)} & x_{2 \ (2)}
\\
&x_{1 \ (3)} & x_{2 \ (3)} & x_{2 \ (3)}
\\
&\vdots & \vdots  & \vdots
\end{array}
\right]
\to \left[
\begin{array}{}
&x_{1 \ (1)} & x_{2 \ (1)} & x_{2 \ (1)} & x_{1 \ (1)}^2
\\
&x_{1 \ (2)} & x_{2 \ (2)} & x_{2 \ (2)} & x_{1 \ (2)}^2
\\
&x_{1 \ (3)} & x_{2 \ (3)} & x_{2 \ (3)} & x_{1 \ (3)}^2
\\
&\vdots & \vdots  & \vdots  & \vdots
\end{array}
\right]
$$
From the point of view of the algorithm to calculate the weight matrix nothing changes, we just need to find one more weight than before.

---
# Regularization Term
Often to compensate after extending the matrix, or for compensate on overfitting, we introduce a regularization term,
so we add to the cost function the term $\mu$ :
$$
E(\hat{X}, w) = \ldots + \mu w^2
$$
So  when calculating the cost, we can impose that the weights do not assume too high values.

**IDEA**:
The regularization term can be think about a **flatting** term, augmenting $\mu$ the resulting plot will be more flat.

![[Pasted image 20220206142718.png]]

---
###### ~Ex.: Solution even with $d+1 \lt l$ 
![[Pasted image 20220206142305.png]]

---
