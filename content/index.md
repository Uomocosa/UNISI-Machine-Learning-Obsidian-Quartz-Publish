# Machine Learning
- Corso del 1° Anno di Magistrale (1° Semestre).
- Docente: **Mario Gori**.
- [Link to Drive with Video Lectures](https://drive.google.com/drive/u/1/folders/1sNEts1eyF2qfiCXhqqL72wmIuolZd_uS)
- [Link to Drive with Other Stuff](https://drive.google.com/drive/u/1/folders/1iFlw2I_OIIsmIGtaoyXlFmlhxVQqODzq2RyckWLJiMtQICUhv-n6LWZ3mxlT5ucO5ZVwYtQl)
										<br>
---
## Perquisites:

---
# Index
[[ML - Lecture 3]]
[[ML - Lecture 4]]
[[ML - Lecture 5]]
[[ML - Lecture 6]]
[[ML - Lecture 7]]
[[ML - Lecture 8]]
[[ML - Lecture 9]]
[[ML - Lecture 10]]
[[ML - Lecture 11]]
[[ML - Lecture 12]]
[[ML - Lecture 13]]
[[ML - Lecture 14]]
[[ML - Lecture 15]]
[[ML - Lecture 16]]
[[ML - Lecture 17]]
[[ML - Lecture 18]]
[[ML - Preparation for the Written Exam]]
[[README Documentation 'General_Backpropagation_NN']]


## Exercises:
[[ML - All Exercises|Discrete Event Systems - All Exercises]]

###### Checkbox Questions:
Which one of the following is a regression problem? 
- [ ] Perform stock market prediction on the basis of a window of previous samples
- [ ] Decide whether a given fingerprint belongs to a given person
- [ ] Predict whether the rank of a given Web page exceeds a given threshold
- [ ] Nothing of the above.
<br>

Which of the following statements concerning the one-hot encoding is correct? 
- [ ] The one-hot encoding corresponds with the traditional binary encoding of integers
- [ ] The one-hot encoding is used only for deep networks; The one-hot encoding is based on one output only
- [ ] The one-hot encoding of n classes consists of n outputs. 
- [ ] The target is null for all outputs apart from the one which encodes the specific class.
<br>

Which one of the following is correct concerning the notion of training set and test set? 
- [ ] Both the sets can be used for the discovery of the weights of the neural network
- [ ] The test set can be used in the learning algorithm only to check overfitting
- [ ] Training and test set are synonyms. 
- [ ] The test set cannot be used in the computation of the weights of the neural network. 
- [ ] Nothing of the above
<br>

Let us consider the quadratic loss function:
$$
V\left(f(x), \ y\right) = \frac{1}{2}\left(f(x) - y\right)^2
$$
Which of the following statements is correct?
- [ ] The loss function is null only if the output of the function fits perfectly the target
- [ ] The loss function can be used for classification but not for regression; When we use sigmoidal units, this loss function cannot be used if the target y does not take values in {−1, +1}
- [ ] Nothing of the above
<br>

Let us consider the loss function
$$
V\left(f(x), \ y\right) = min\left\{0.1 - y(x)f(\omega, x)\right\}^2
$$
Which of the following statements is reasonable? 
- [ ] The loss function is adequate for classification
- [ ] The loss function is adequate for regression
- [ ] The is not a loss function since it is not differentiable in all points of its domain
- [ ] Nothing of the above.
<br>

What is the difference between loss function and empirical risk function? 
- [ ] They are synonyms
- [ ] The loss function refers to the error on single examples, whereas the risk function refers to the error over all the examples of the training set
- [ ] The loss function is always differentiable whereas the empirical risk function may not be differential
- [ ] Nothing of the above.
<br>
Which one of the following are regression problems? 
- [ ] Decide when to buy and when to sell on the stock market on the basis of a window of previous samples
- [ ] Decide whether two fingerprints belong to the same person
- [ ] Predict the annual income of a company on the basis of the field of business and on the number of employees
- [ ] Nothing of the above.
<br>

What is the meaning of overfitting? 
- [ ] It is a synonym of “best fitting”
- [ ] It is refers specifically to the LMS algorithm, for the case of quadratic loss
- [ ] It indicates a fitting of the training set with scarse degree of parsimony
- [ ] Nothing of the above
<br>

Which one of the following is correct concerning the saturation of sigmoidal neurons? 
- [ ] Sigmoidal neurons saturates when the value of the weights become big
- [ ] Sigmoidal neurons never saturates
- [ ] The saturation of sigmoidal neurons is independent of the input
- [ ] Nothing of the above.
<br>

Let us consider the supposed loss function 
$$
V (f(x), y) = −y  \operatorname{log}(f) \  − (1 − y) \operatorname{log}(1 − f) \kern 30px \text{where} \ y \in \{0, 1\} 
$$
is the target and f is the value returned by a sigmoidal neuron in the scalar case. Which of the following holds true? 
- [ ] This is an entropy, but it is not a loss function since it returns negative values
- [ ] This loss function is typically better than the quadratic loss for classification
- [ ] The above entropy loss can also be used with targets in {−1, +1}
- [ ] Nothing of the above
<br>

Let us consider the empirical risk function
$$
E = \left(\sum_{k = 1}^{l} (h_k - j(\omega_k, \ x_k))^{2m}\right)^{1/2m} \kern 30 px \text{where} \ m \in \mathbb{N}
$$
Which of the following statements is true?
- [ ] The learning with this empirical risk always returns a perfect match $E \to 0$ for $m \to \infty$
- [ ] This empirical risk returns the maximum error $max_k | (y_k - f(\omega, \  x_k)) |$ independently of learning as $m \to \infty$
- [ ] Nothing of the above
<br>

Which of the following statements is true concerning the regualarization parameter in ridge regression?
- [ ] The regularization parameter can be any small real number
- [ ] The regularization parameter leads to discover a unique solution in normal equations
- [ ] The regularization parameter improves the fitting on the training set
- [ ] There is always a unique solution in normal equation also if the regularization parameter is zero
- [ ] Nothing of the above
<br>

###### Open Questions:
- Discuss the following statements concerning the recognition performance of neural networks for handwritten chars.
	- **a)** If we significantly increase the 28 × 28 MNIST resolution we expect to increase significantly the recognition performance
	- **b)** Pictures of handwritten digits that can be collected with ordinary smartphones have a resolution which is significantly higher than 28 × 28. Can you still see any other reason for keeping a limited resolution in the experiments with neural nets?
	- **c)** Suppose you have trained successfully a neural network on the MNIST database and you want to write an application which recognizes digits by using your own smartphone. Describe a pre-processing algorithm that uses the neural network trained on MNIST for recognizing the digits given by pictures taken on your smartphone at higher resolution.
<br>

- Discuss a neural network - by explicitly indicating the values of the weights - that composed of the cascade of two rectified with the purpose of realizing the function shown in the following figure:
![[Pasted image 20220205175054.png]]
<br>

- Consider the Boolean function:
$$
f(x,y,z) = x \wedge y \wedge z
$$
Is it lienearly separable? Proof is required
<br>

- Suppose you are given a multilayared neural network with two inputs, one output and any number $p$ of arbitrarily large hidden layers. If the neurons are linear, that is:
$$
x = \sigma(a) = a
$$
can this neural network compute the XOR predicate? Motive the answer.
<br>

- Consider a collection of black & white pictures and suppose we want to separate those with more black than withe pixels.
Can we solve this problem by a neural network with one sigmoidal neuron only? Proof is required.

---

## Python Scripts:

----
###### All My Notes
For the best experience in reading these and all other notes, and also if you wish to EDIT them, do as follows: 
1. Install [Obsidian](https://obsidian.md), or another markdown editor.
2. Go to the Github link of this or another note
3. Download all the repo or if you know git just the 'content/' folder
4. Extract just the 'content/' folder from the repo zip file
5. Open Obsidian >> Menage Vaults >> Open Folder as Vault >> and select the 'content/' folder you just extracted

==PLEASE NOTE==:
- These notes were not revised by the professors, so take all of them with a grain of salt.
- However if you download them since they are made in markdown you can EDIT them, please do so.
- If you edit and "upgrade" them, please pass the new ones to the other students and professors.

Here are all the links to my notes:
- ***Github***: [UNISI-Sensors-and-Microsystems-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Sensors-and-Microsystems-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Sensors-and-Microsystems-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Sensors-and-Microsystems-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Complex-Dynamic-Systems-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Complex-Dynamic-Systems-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Complex-Dynamic-Systems-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Complex-Dynamic-Systems-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Discrete-Event-Systems-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Discrete-Event-Systems-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Discrete-Event-Systems-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Discrete-Event-Systems-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-System-Identification-and-Data-Analysis-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-System-Identification-and-Data-Analysis-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-System-Identification-and-Data-Analysis-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-System-Identification-and-Data-Analysis-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Multivariable-NonLinear-and-Robust-Control-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Multivariable-NonLinear-and-Robust-Control-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Multivariable-NonLinear-and-Robust-Control-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Multivariable-NonLinear-and-Robust-Control-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Artificial-Intelligence-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Artificial-Intelligence-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Artificial-Intelligence-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Artificial-Intelligence-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Human-Centered-Robotics-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Human-Centered-Robotics-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Human-Centered-Robotics-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Human-Centered-Robotics-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Machine-Learning-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Machine-Learning-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Machine-Learning-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Machine-Learning-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Bioinformatics-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Bioinformatics-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Bioinformatics-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Bioinformatics-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Network-Optimization-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Network-Optimization-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Network-Optimization-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Network-Optimization-Obsidian-Quartz-Publish).
- ***Github***: [UNISI-Mathematical-Methods-for-Engineering-Obsidian-Quartz-Publish](https://github.com/Uomocosa/UNISI-Mathematical-Methods-for-Engineering-Obsidian-Quartz-Publish);<br>***Quartz Publish***: [UNISI-Mathematical-Methods-for-Engineering-Obsidian-Quartz-Publish](https://uomocosa.github.io/UNISI-Mathematical-Methods-for-Engineering-Obsidian-Quartz-Publish).
