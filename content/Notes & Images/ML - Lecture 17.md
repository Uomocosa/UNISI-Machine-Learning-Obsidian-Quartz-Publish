# Alternatives to Backpropagation
Remembering that the updated of the weights is done by
$$
w \leftarrow w - \eta \cdot \nabla E
$$
And $\nabla E$ is nothing more then:
$$
\frac{e(w_{ij} + h) - e(w_{ij})}{h}
$$
we can empirically calculate it if we take a really small $h$.

**But is it any good ?**
Actually **no**, the backpropagation algorithm ha complexity $O(m)$ while empirically calculating the derivative has complexity $O(m^2)$ 

This is due to the fact that taking the length of the NN ($m$) we will need to run only one feedforward algorithm for calculating the backpropagation one

While for calculating the error you need to re-run the feedforward propagation for each feedforward step, (calculate all errors for $w_{ij}$).