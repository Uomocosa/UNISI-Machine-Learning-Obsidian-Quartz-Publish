# Terminology:
**ANN**: Artificial NN
**NN**: Neural Network
**MNIST**: Online database of handwritten characters (resolution usually of $28 \times 28 = 784$ pixel)

---
# Structure of a NN
A simple NN that gets in input $784$ pixel from a $28 \times 28$ MNIST Handwritten Character where each input assume values in $[0, 255]$ where $0$ corresponds to complete white and $255$ to complete black.

If we use a simple NN with no hidden layer, and we know that the handwritten characters are only numbers from 0 to 9, we can have a NN with 
**INPUT LAYER**: 784 features + 1 bias 
**OUTPUT LAYER**: 10 classes

![[Pasted image 20220206175853.png]]
 
We have to find $785 \times 10$ weights that solve this problem

Now suppose we use the sigmoid function as the activation function.
The sigmoid function assume values $\in [0, 1]$, the output values of the NN (10 outputs) will be given by this function, so to choose the final output of the NN, we choose the highest value across all 10 outputs.

![[Pasted image 20220206175927.png]]

---
# Loss Function of Sigmoid NN
From the NN given before we can define its Loss function or Error function as follows:
$$
\begin{align}
E = \frac{1}{2} \sum_{k = 1}^l \left(y_k - \sigma\left(\hat{w}^T \kern 1px \hat{x}_k\right)\right)^2
\\[5px]
\hat{w}^* = {\operatorname{argmin}\atop{\small \hat{w}}} \ E(\hat{w})
\end{align}
$$
So we have that its partial derivative with respect to $w$ is:
![[Pasted image 20220206180128.png]]

And the partial derivatives with respect to $b$ is:

Together they form the nabla of E ($\nabla E$) that can be used to update the weight:
$$
\hat{w} \leftarrow \hat{w} - \eta \kern 3px \nabla E
$$
---
# Sign Rule
$E(w)$ and $\nabla E$ will always have the same sign.