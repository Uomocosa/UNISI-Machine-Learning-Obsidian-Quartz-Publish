###### Study the evolution of the regression algorithm (ridged and non)
###### Study the evolution of the classification algorithm
###### Emphasis on the formula used

# Backpropagation:
**Error** or **Risk** function: $E(X, \Theta) = \frac{1}{2N}\sum_{i=1}^N\left(\hat{y}_i - y_i\right)^2$

Updating the weights: $w_{ij}^k \leftarrow w - \alpha \frac{\partial E(X, \Theta)}{\partial w_{ij}^k}$

In case of activation function being the sigmoid function: $\sigma(a^k) = \frac{1}{1 + e^{-a^k}}$

Then the specific update of the weights becomes:
- For the weights on the last layer $N$:
$$
\frac{\partial E}{\partial w^N} = \operatorname{mean}\left(\sigma\left(a^{N-1}\right) \cdot \sigma\left(a^N\right) - y\right)
$$
- For all the others:
$$
\frac{\partial E}{\partial w^k} = \operatorname{mean}\left(
	\sigma\left(a^{k-1}\right)
	\cdot 
	\left(1 - \sigma\left(a^{k}\right)\right) 
	\cdot
	w^{k+1}\frac{\partial E}{\partial w^{k+1}}
\right)
$$
Remember that the vector $a^0 = x_0$.

