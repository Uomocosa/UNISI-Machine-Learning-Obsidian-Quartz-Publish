# Pseudo-Inversion
From the formula $(\hat{X}^T\kern 1px\hat{X})\kern 1px\hat{W}^*$ what if $(\hat{X}^T\kern 1px\hat{X}) \in \mathbb{R}^{d+1, \ d+1}$ is not invertible.
Maybe $(\hat{X}\kern 1px\hat{X}^T) \in \mathbb{R}^{l, \ l}$ is.

That is because the number of features, plus the bias ($d +1$) is really often much less than the number of examples taken ($l$)

So i can change the formula a little:
$$
(\hat{X}^T\kern 1px\hat{X})\kern 1px\hat{W}^* = \hat{X}\kern 3px Y \to \hat{X} \kern 1px \hat{W} = Y 
$$
So (pseudo inversion):
$$
\hat{W}^* = \hat{X}^T \kern 1px \left(\hat{X} \kern 1px \hat{X}^T\right)^{-1} \kern 3px Y
$$
And:
$$
\begin{align}
\hat{X} \kern 1px \hat{W}^* &= \hat{X} \kern 1px \hat{X}^T \kern 1px \left(\hat{X} \kern 1px \hat{X}^T\right)^{-1} \kern 3px Y 
\\
&= Y
\end{align}
$$
---
# Gradient Descent
![[Pasted image 20220206110455.png]]

**IDEA**:
Given the [[ML - Lecture 4#Linear Neuron|nabla of the error function]] $\nabla E$: we can calculate the next "step" such that updating the weights $w$ will bring us to a smaller $E$.

**REMBER**:
Ur objective is to bring $E \to 0$.

**ADVANTAGES**:
Instead of using the "exact" formula to calculate the weights we can use gradient descent.
Gradient descent will find a solution *even if one perfect solution does not exist*.

---
### Updating Parameter $\eta$ :
$\eta$ defines how big of a step we will take.
- Bigger $\eta$ can solve the problem faster (less step required) but can also not solve the problem at all $E \to \infty$ instead of going to 0
- Smaller $\eta$ will bring a more accurate solution but it will take longer

---
