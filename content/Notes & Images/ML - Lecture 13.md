###### How many steps do a general NN take to find a solution?
Starting from the formula for updating the weights:
$$
w \leftarrow w - \eta \cdot \nabla E
$$
We consider the updating process **continuous** and say:
$$
\frac{\partial w}{\partial t} = -\eta \cdot \nabla E
$$
Now suppose we use a new learning rate:
$$
\eta \rightarrow \frac{\eta}{\parallel\nabla E\parallel^2}
$$
The idea behind this is that when we are in a platò ($\frac{\partial E}{\partial w} = 0$) the learning rate becomes huge, to reduce the step needed to exit the platò, while we are in a deep slope the learning rate automatically reduces to "proceed with caution"

We choose the square ($\parallel\nabla E\parallel^2$) because: 
$$
\begin{align}
\frac{\partial E}{\partial t} &= \frac{\partial}{\partial t}E(w(t))
\\[5px]
&= \nabla E \cdot \frac{\partial w}{\partial t}
\\[5px]
&= \nabla E \cdot \left(-\eta \frac{\nabla E}{\parallel\nabla E\parallel^2}\right)
\\[5px]
&= -\eta \kern 3px \left(\frac{\nabla E \cdot \nabla E}{\parallel\nabla E\parallel^2}\right)
\\[5px]
&= -\eta \kern 3px \left(\frac{\parallel\nabla E\parallel^2}{\parallel\nabla E\parallel^2}\right)
\\[5px]
&= -\eta
\end{align}
$$
So the error decreases **linearly** with $\frac{\partial E}{\partial t} = - \eta$.

Given $E_0 = E(t = 0)$ we can expect the error at time $t$ to be: $E_0 - \eta \kern 2px t$, so if we want to know ho much time $t^*$ is required to bring the error to $0$:
$$
t^* = \frac{E_o}{\eta}
$$
---
### Solution of non-linear separable problems with NN
~ XOR function
- The XOR function is not linearly separable
- A NN with sigmoid or sign activation function, and an **hidden layer** can solve it.

**TODO:** Make example of a NN that solves the XOR function