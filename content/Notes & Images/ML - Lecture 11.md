# Example of Chain Rule
Given the chain rule:
$$
\frac{\partial}{\partial w_i}e_k = \frac{\partial V}{\partial f}\frac{\partial f}{\partial w_i}
$$
In the case of NN with sigmoid activation function we have
$V\left(f\left(\hat{w}, \hat{x}_k\right), y_k\right)$
$$
f\left(\hat{w}, \hat{x}_k\right) = \sigma\left(\hat{w}, \hat{x}_k\right)
$$
So its partial derivative with respect to $w$ is:
$$
\frac{\partial f}{\partial \hat{w}_i} = \frac{\partial \sigma(a_k)}{\partial \hat{w}_i}\hat{x}_{k_{\Large i}}
$$
Also, Given:
$$
V = \frac{1}{2}\left(f\left(\hat{w}, \hat{x}_k\right), y_k\right)^2 = \frac{1}{2}\left(\sigma(a_k) - y_k\right)^2
$$
We have that:
$$
\frac{\partial V}{\partial f} = \left(\sigma(a_k) - y_k\right)
$$
So:
$$
\begin{align}
\frac{\partial V}{\partial w_i} &= 
\\[5px]
&= \frac{\partial V}{\partial f}\frac{\partial f}{\partial w_i} 
\\[5px]
&= \left(\sigma(a_k) - y_k\right) \cdot \frac{\partial \sigma(a_k)}{\partial \hat{w}_i}\hat{x}_{k_{\Large i}}
\end{align}
$$
We also know that:
$$
\frac{\partial \sigma(a_k)}{\partial \hat{w}_i} = \sigma(a_k)\cdot\left(1-\sigma(a_k)\right)
$$
To help us with notations we define the **delta error**:
$$
\delta_k := \frac{\partial \sigma(a_k)}{\partial \hat{w}_i} (\sigma(a)-y_k)
$$
So:
$$
\frac{\partial V}{\partial w_i} = \delta_k \cdot \hat{x}_{k_{\Large i}}
$$
---
# Backpropagation Formula:
The main formula to remember for the backpropagation algorithm using a sigmoidal NN is: 
$$
\frac{\partial V}{\partial w_i} = \sigma(a_k) \cdot \left[1-\sigma\left(a_k\right)\right] \cdot \left[\sigma\left(a_k\right) - y_k\right] 
$$
---
# One-hot Encoding
[Source](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)
**LABEL ENCODING** (Look at the Categorical value column)
```
╔════════════╦═════════════════╦════════╗ 
║ CompanyName Categoricalvalue ║ Price  ║
╠════════════╬═════════════════╣════════║ 
║ VW         ╬      1          ║ 20000  ║
║ Acura      ╬      2          ║ 10011  ║
║ Honda      ╬      3          ║ 50000  ║
║ Honda      ╬      3          ║ 10000  ║
╚════════════╩═════════════════╩════════╝
```

**ONE-HOT ENCODING**:
```
╔════╦══════╦══════╦════════╦
║ VW ║ Acura║ Honda║ Price  ║
╠════╬══════╬══════╬════════╬
║ 1  ╬ 0    ╬ 0    ║ 20000  ║
║ 0  ╬ 1    ╬ 0    ║ 10011  ║
║ 0  ╬ 0    ╬ 1    ║ 50000  ║
║ 0  ╬ 0    ╬ 1    ║ 10000  ║
╚════╩══════╩══════╩════════╝
```

We usually prefer one-hot encoding in respect to categorical value for 2 main reasons
- The label encoding assumes **hierarchy**, if our ML Model internally calculates the average then if we use label encoding we have that: _VM_ < _Acura_ < _Honda_, which doesn't make any sense.
- The one-hot encoding can also be compared to the output of a sigmoid function, or any other ML activation function that output belongs $\in [0, 1]$.

---
# Entropy Loss
$$
\begin{align}
&&V &= -y \log(\sigma(a)) - (1-y)\log(1-\sigma(a)) &= \left\{
	\begin{array}{cc}
	\log(1-\sigma(a)) & \text{for} \ y = 0
	\\
	\log(\sigma(a))& \text{for} \ y = 1
	\end{array}\right.
\\[5px]
&&\frac{\partial V}{\partial a} &= y(1-\sigma(a)) + (1-y)(\sigma(a)) &= \left\{
	\begin{array}{cc}
	1 - \sigma(a) & \text{for} \ y = 0
	\\
	\sigma(a)& \text{for} \ y = 1
	\end{array}\right.
\end{align}
$$

**REMEMBER**:
For classification problems $y$ can only be $0$ or $1$

**OBSERVATION**:
For the Entropy loss the delta error is null only for **absolute minima**

**OBSERVATION**:
If the neuron is saturated ($\sigma(a) = 0 \ \text{or} \ 1$) but the actual output is the opposite ($y = 1 \ \text{or} \ 0$) the entropy returns a big value, think of it as notifying the NN that it made a big mistake.
When the loss is big loss the next step made by the NN will also be big, so with the Entropy loss is much easier to **escape** the condition of **saturation**

