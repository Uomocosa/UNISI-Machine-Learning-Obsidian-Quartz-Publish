# Recap:
*Backpropagation*:
- **Batch-mode** (all examples are first summed to update the weights, for each epoch the weights are updated only 1 time)
- **Mini-batches** (some examples are first summed to update the weights, for each epoch the weights are updated a determined number of times)
- **Online-mode** (only one example is used to update the weights, there are no epochs)

**Initialization of the Weights**:
There is a risk (especially with sigmoidal functions) that if you are using big weights the cost function $V$ becomes too big and saturates, a good rule is to use weights $\in [0, 1$].
Also it's important to notice that the weights should be initialized at random, especially not setting them at $0$, this is because, as explained on [Coursera](https://www.coursera.org/learn/machine-learning/lecture/ND5G5/random-initialization) all the weights multiplying the same variable will remain the same, no matter how many times we iterate the backpropagation algorithms.

**Saturation**:
Pay particularly attention to this problem, because one saturated layer can increase the learning time required by many folds.
To get around this problem you could use a ReLU or even better a Leaky-ReLU
Using big networks decreases the possibility of saturation, but not always a big network is the solution (they are more slow for sure)

**Saturation of a ReLU**
The interesting path of a saturated ReLU is that its output will be 0, so this means that if saturated it will block the flowing of information.

**Normalization of the Inputs**
Before passing the inputs to the NN you can apply one "normalization layer" that divides the inputs for their mean value (based on all the examples)
Or instead of the mean it divides them by a factor: $(\text{max\_value} - \text{min\_value})$, even if this is more efficient can result in some problem if the variance of a parameter is too big 
~ For example: $\text{min\_value = 1}$, $\text{max\_value = 100000}$ then if i take a value like $2$ or $10$, after the normalization layer for a machine the value could be the same (remember: a computer does not visualize all numbers), tho this is kinda rare, take in account that it could happen.

---
# No Local Minima
Even big network do not get stuck in local minima and generalize well.

The intuition here is that in a bilion-parameters NN there will always be a path or a variable that changed will decrease the Error, this will bring us to at least a sub-local minima (a good approximation even if not the global minima)