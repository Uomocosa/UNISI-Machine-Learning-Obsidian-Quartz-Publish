# Rosenblatt's Perceptron Algorithm
Given a training set $\mathcal{L}$ with targets $y_i$ taking values $\pm 1$, find $\hat{w}$ and $t$ such that the hyperplane perpendicular to $\hat{w}$ correctly separates the examples and $t$ is the number of times that $\hat{w}$ is updated.
1. **INITIALIZE**: Set $\hat{w}_0 \leftarrow 0$, $t \leftarrow 0$, $j \leftarrow 1$, and $m \leftarrow 0$.
2. **NORMALIZE**: Compute $R$ for all $i = 1, \ \ldots, \ l$ set $\hat{x}_i \leftarrow (\hat{x}_i, R)^T$.
3. **CARROT OR STICK ?**: If $y_j\kern3px\hat{w}^T\kern0px\hat{x}_j \le 0$, set $\hat{w} \leftarrow \hat{w} + \eta \kern 2px y_j \hat{x}_j$, $t \leftarrow t + 1$, $m \leftarrow m + 1$.
4. **ALL TESTED ?**: Set $j \leftarrow j + 1$; If $j \neq l$ go back to step 3.
5. **NO MISTAKES ?**: If $m = 0$, the algorithm terminates; set $\hat{w} \leftarrow (w, b/R)$ and return $(\hat{w}, t)$.
6. **TRY AGAIN**: Set $j \leftarrow 1$, $m \leftarrow 0$, and go back to step P3.


> $y_j\kern3px\hat{w}^T\kern0px\hat{x}_j \le 0$ : because $y_i$ can only be $-1$ or $+1$, (two classes), the classification in this case is defined as **sign agreement**, because the supervisor stop the algorithm only when the sign agrees ($y_j\kern3px\hat{w}^T\kern0px\hat{x}_j \gt 0$)

> This algorithm can also be seen as a NN where the activation function is the _sign_ function

---
# ReLU
**Re**ctified **L**inear **U**nit
![[Pasted image 20220207114200.png]]

Another activation function that can be substituted to the _sigmoid_ function.

Prevents saturation for $a > 0$ but not for $a < 0$.

Also note that the derivate for $a = 0$ doesn't exist.
So we have to directly specify its value in the code (not too difficult)

---
### Robust Linear Separation
The Rosenblatt's perceptron algorithm perform a **robust** linear separation

Robust because all the points are divided from the linear separation by a factor of $\delta$ (distance), this value is not known at prior, it can be found as the $\min$ of the distances from the line of linear-separation and all the points.

![[Pasted image 20220207114709.png]]

Also from calculations and theorems we get that the number of steps $t$ that allow the algorithm to find the perfect solution will be:
$$
t \le 2 \left(\frac{R}{s}\right)^2
$$
Where R is the radius of the space occupied by the points, as shown in the figure.

---
### Linear Separation with more variables
Let's now get an intuition on why having more features can increment the possibility of finding linear-separation in the data.

Do do this let's see to the opposite case.

Take 3 point in a 2D plane:
![[Pasted image 20220207115604.png]]
![[Pasted image 20220207115624.png]]

All of this can be linearly separated by a single line
Also even if we project them in a 1D plane they can still be linearly separated:

First one for example:
![[Pasted image 20220207115812.png]]

![[Pasted image 20220207115746.png]]
![[Pasted image 20220207115737.png]]
So this can all be linearly separated

Now take for example this points and their projection:
![[Pasted image 20220207120224.png]]![[Pasted image 20220207120239.png]]

**Notice** how the 3 point in the 2D plane can be linearly separated, while the points projected in the 1D plan **cannot**

