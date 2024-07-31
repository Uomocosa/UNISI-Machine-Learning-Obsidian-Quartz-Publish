# MNIST Example:
Starting with an example:
After downloading the MNIST training and test data set, from the [official site](http://yann.lecun.com/exdb/mnist/), I read the `.gz` files and convert them in NN friendly numpy.arrays, using the functions [[#get_numpy_matrices_from_MNIST_set]] and [[#get_numpy_array_from_MNIST_labels]].
Then using my custom class [[#General_Backpropagation_NN]] I created a function that creates an empty NN, [[#create_NN]], where there are defined only the cost function and activation function that i want to use, for this example I opted for the cost function: [[#cross_entropy_loss]] and the activation function: [[#ReLU]]
> [[#How to create a new Cost Function]]
> [[#How to create a new Activation Function]]

I had to create first an empty NN first because the MNIST dataset provides two different datasets one for training and one for testing.

Then after using the one hot bit encoding i created 2 dictionaries, the first that converts the labels in the one bit encoding and the other dictionary that does the opposite.
> [[#Dictionaries for one hot bit encoding]]

So i classified the labels using the function [[#classify_label]] according to the one bit encoding dictionary just created

Lastly, I create the NN and set manually the test set, validation set, and test set:
```python
NN = create_NN(
		cost_function = mp.Machine_Learning.Cost_Functions.cross_entropy_loss,
		activation_function = mp.Machine_Learning.Activation_Functions.ReLU
	)

m = NN.number_of_examples = training_set[0].shape[0]

ceil = math.ceil
	NN.validation_set = tuple((training_set[0][ceil(m*0.8):], training_set[1][ceil(m*0.8):]))
	NN.training_set = tuple((training_set[0][:ceil(m*0.8)], training_set[1][:ceil(m*0.8)]))
	NN.test_set = test_set
```

For context we start with:
`STARTING ACCURACY ON TEST SET: 7.6499999999999995%`

And after training for $100$ *epochs* with a *learning rate* of $0.001$ using this function
```python
NN.train_for_n_epochs(
	n_epochs = 100, 
	learning_rate = 0.001, 
	stop_execution = False, 
	show_plot = True
)
```

We end with an accuracy of:
`ACCURACY ON TEST SET: 11.3%`

I tried incrementing the n_epochs to $10000$ and the accuracy incremented to around $30\%$

---
# General_Backpropagation_NN
I class to create a simple backpropagation Neural network, given labeled_inputs and labeled_ouptuts.
By default the NN uses the cost function: *logistic_loss*, and the activation function: *sigmoidal*, but they can be changed.
> [[#How to create a new Cost Function]]
> [[#How to create a new Activation Function]]

Also it is possible to create a stopping criteria based on the errors of training and validation set during training.
> [[#How to create a new Stopping Criteria]]

Already in the package there are some useful functions that can be used for the creation of the NN:
- Activation Functions:
	- [[#hyperbolical]]
	- [[#ReLU]]
	- [[#sigmoidal]]
	- [[#sinusoidal]]
- Cost_Functions:
	- [[#cross_entropy]]
	- [[#logistic_loss]]
	- [[#min_square_loss]]
	- [[#simple_loss]]
- Stopping Criteria
	- [[#same_after_1000_iterations]]
	- [[#same_after_1000_iterations_variation]]


The code for the class is divided in:
- Class Initialization: [[#__init__]]
- Methods:
	- [[#create_random_THETA]]
	- [[#feedforward_propagation]]
	- [[#backpropagation]]
	- [[#calculate_error]]
	- [[#train_for_one_epoch]]
	- [[#train_for_n_epochs]]
	- [[#evaluate]]

---
# Activation Functions:
### How to create a new Activation Function
All function will receive the same inputs:
- `a` : dictionary with `a[0]` : inputs, `a[1]`: first layer ouputs, `a[2]`: second layer ouputs, ...
- `w` : dictionary with `w[1]` : weights first layer, `w[2]`: weights second layer, ...
- `b` : dictionary with `b[1]` : biases first layer, `b[2]`: biases second layer, ...
- `i` : index

The output of the function '`formula`' corresponds to the output of the layer `i`
$$
a^{[i]} = \sigma(a^{[i-1]} \cdot w^{[i-1]} + b^{[i-1]})
$$

The output of the function '`derivate`' corresponds to:
$$
\frac{\partial \sigma(z)}{\partial z}
$$

> **NOTE**:
> A new Activation Function file is expected to have 2 functions one called `formula` and the other one `derivate`

---
### hyperbolical
```python
# a = sigma(z)
# define here sigma
def formula(a, w, b, i):
	ones_vector = numpy.ones((1, a[i-1].shape[0]))
	z = a[i-1] @ w[i].T + (b[i] @ ones_vector).T
	return numpy.tanh(z)




def sech(x): return 1/numpy.cosh(x)

# a = sigma(z)
# define here sigma'(z)
def derivate(a, w, b, i):
	ones_vector = numpy.ones((1, a[i-1].shape[0]))
	z = a[i-1] @ w[i].T + (b[i] @ ones_vector).T
	return sech(z)**2
```

---
### ReLU
```python
# a = sigma(z)
# define here sigma
def formula(a, w, b, i):
	ones_vector = numpy.ones((1, a[i-1].shape[0]))
	z = a[i-1] @ w[i].T + (b[i] @ ones_vector).T
	z[z <= 0] = 0
	return z




# a = sigma(z)
# define here sigma'(z)
def derivate(a, w, b, i):
	ones_vector = numpy.ones((1, a[i-1].shape[0]))
	z = a[i-1] @ w[i].T + (b[i] @ ones_vector).T
	dz = numpy.ones(z.shape)
	dz[z <= 0] = 0
	return dz
```

---
### sigmoidal
```python
# a = sigma(z)
# define here sigma
def formula(a, w, b, i):
	ones_vector = numpy.ones((1, a[i-1].shape[0]))
	z = a[i-1] @ w[i].T + (b[i] @ ones_vector).T
	return ML.sigmoid(z)




# a = sigma(z)
# define here sigma'(z)
def derivate(a, w, b, i):
	ones_vector = numpy.ones((1, a[i-1].shape[0]))
	z = a[i-1] @ w[i].T + (b[i] @ ones_vector).T
	return ML.sigmoid(z) * (1 - ML.sigmoid(z))
```

---
### sinusoidal
```python
# a = sigma(z)
# define here sigma
def formula(a, w, b, i):
	ones_vector = numpy.ones((1, a[i-1].shape[0]))
	z = a[i-1] @ w[i].T + (b[i] @ ones_vector).T
	return numpy.sin(z)




# a = sigma(z)
# define here sigma'(z)
def derivate(a, w, b, i):
	ones_vector = numpy.ones((1, a[i-1].shape[0]))
	z = a[i-1] @ w[i].T + (b[i] @ ones_vector).T
	dz = numpy.ones(z.shape)
	dz[z <= 0] = 0
	return numpy.cos(z)
```

---
# Cost Functions:
### How to create a new Cost Function
Like for the activation functions all cost functions will receive the same inputs:
- `y` : numpy.ndarray with the real outputs
- `y_hat` : numpy.ndarray with the evaluated outputs given by the NN

Define in a function how the cost is computed, under some examples:

---
### cross_entropy_loss
```python
def cross_entropy_loss(y, y_hat):
	loss = numpy.zeros(y.shape)
	assert numpy.any([y != 0] and [y != 1])
	assert numpy.all([y_hat >= 0])
	loss[y==1] = - myNP.log(numpy.absolute(y_hat[y==1]))
	return loss
```

---
### logistic_loss
```python
def logistic_loss(y, y_hat):
	loss = numpy.zeros(y.shape)
	assert numpy.any([y != 0] and [y != 1])
	assert numpy.all([y_hat >= 0])
	loss[y==0] = - myNP.log(numpy.absolute(1 - y_hat[y==0]))
	loss[y==1] = - myNP.log(numpy.absolute(y_hat[y==1]))
	return loss
```

---
### simple_loss
```python
def simple_loss(y, y_hat):
	return numpy.absolute(y - y_hat)
```

---
### min_square_loss
```python
def min_square_loss(y, y_hat):
	return (y - y_hat)**2
```

---
# Stopping Criteria
### How to create a new Stopping Criteria
The inputs for a stopping criteria function are:
- `J_training_history` : tuple with all the costs relative to the training set as the training of the NN progresses
- `J_validation_history` : tuple with all the costs relative to the validation set

The function is expected to return a bool:
- `True` if the criteria is met and the training of the NN needs to stop
- `False` otherwise

---
### no_stopping_criteria
```python
def no_stopping_criteria(J_training_history, J_validation_history):
	return False
```

---
### same_after_1000_iterations
```python
def same_after_1000_iterations(J_training_history, J_validation_history):
		if len(J_training_history) < 1000: return False
		J_validation_history = numpy.array(J_validation_history[-1000:])
		J_validation_history -= J_validation_history[0]
		if numpy.all(numpy.absolute(J_validation_history) <= 0.01): return True
		else: return False
```

---
### same_after_1000_iterations_variation
The only difference with the stopping criteria above is that the last cost of the training set has to be greater than the last cost of the validation set
```python
def same_after_1000_iterations_variation(J_training_history, J_validation_history):
		if len(J_training_history) < 1000: return False
		if J_training_history[-1] > J_validation_history[-1]: return False
		J_validation_history = numpy.array(J_validation_history[-1000:])
		J_validation_history -= J_validation_history[0]
		if numpy.all(numpy.absolute(J_validation_history) <= 0.01): return True
		else: return False
```


---
 # Function used:

### \_\_init\_\_
Arguments:
- **\[Required\]**: `labeled_inputs` : numpy.array
- **\[Required\]**: `labeled_outputs` : numpy.array
- *(Optional)*: `hidden_layers_sizes` : tuple
- (Optional): `cost_function` : function
- (Optional): `activation_function` : function
- (Optional): `activation_function_formula` : function
- (Optional): `activation_function_derivate` : function
- (Optional): `activation_function_derivate` : function
- (Optional): `THETA` : dict
- (Optional): `range_for_THETA_init` : tuple


Jobs of \_\_init\_\_:
- Assert that all the types of the arguments corresponds.
- Assign default values if some are not passed.
- Create a Random THETA Matrix.

> **NOTE**:
> The THETA is not exactly a Matrix, it's a dictionary of matrices where each item corresponds to the matrix $\bar{\Theta}^{[i]}$ containing the weights and the bias of the layer $[i]$


```python
def __init__(self, 
		labeled_inputs, labeled_outputs, hidden_layers_sizes = tuple(),
		*,
		cost_function = logistic_loss,
		activation_function = sigmoidal,
		activation_function_formula = None,
		activation_function_derivate = None,
		THETA = None,
		range_for_THETA_init = (-10, +10),
	):
		assert type(labeled_inputs) is numpy.ndarray
		assert type(labeled_outputs) is numpy.ndarray
		assert type(hidden_layers_sizes) is tuple
		assert type(range_for_THETA_init) is tuple
		assert labeled_inputs.shape[0] == labeled_outputs.shape[0]

		m = labeled_inputs.shape[0]

		self.training_set = (labeled_inputs[:round(m*0.6)], labeled_outputs[:round(m*0.6)])
		self.validation_set = (
			labeled_inputs[round(m*0.6) + 1:round(m*0.8)],
			labeled_outputs[round(m*0.6) + 1:round(m*0.8)]
		)
		self.test_set = (labeled_inputs[round(m*0.8)+1:], labeled_outputs[round(m*0.8)+1:])

		self.number_of_examples = m
		self.cost_function = cost_function
		if activation_function_formula is None:
			self.activation_function_formula = activation_function.formula
		else: self.activation_function_formula = activation_function_formula
		if activation_function_derivate is None:
			self.activation_function_derivate = activation_function.derivate
		else: self.activation_function_derivate = activation_function_derivate

		if THETA is None:
			self.THETA = self.create_random_THETA(
				input_shape = labeled_inputs.shape[1],
				hidden_layers_sizes = hidden_layers_sizes,
				output_shape = labeled_outputs.shape[1],
				range_for_THETA_init = range_for_THETA_init,
			)
		else: self.THETA = THETA
```

---
### create_random_THETA
Given the `input_shape`, `output_shape` and `hidden_layers_sizes`, as well as an *(Optional)* parameter `range_for_THETA_init` it returns the THETA dictionary.

```python
def create_random_THETA(
	NN, 
	input_shape : int,
	hidden_layers_sizes : tuple, 
	output_shape : int,
	range_for_THETA_init = (-10,10)
):
	THETA = dict()

	if len(hidden_layers_sizes) >= 1:
		number_of_cols = input_shape + 1
		for i in range(len(hidden_layers_sizes)):
			number_of_rows = hidden_layers_sizes[i]
			THETA[i+1] = myNP.create_random_matrix_of_size(
				number_of_rows, number_of_cols,
				range_of_values = range_for_THETA_init,
			)
			number_of_cols = hidden_layers_sizes[i] + 1

		number_of_rows = output_shape
		THETA[i+2] = myNP.create_random_matrix_of_size(
			number_of_rows, number_of_cols,
			range_of_values = range_for_THETA_init,
		)

	else:
		number_of_rows = output_shape
		number_of_cols = input_shape + 1
		THETA[1] = myNP.create_random_matrix_of_size(
			number_of_rows, number_of_cols,
			range_of_values = range_for_THETA_init,
		)

	return THETA
```

---
### feedforward_propagation
Basic formula of feedforward_propagation, given some `inputs` it returns a dictionary with the layers outputs (`a`).

```python
def feedforward_propagation(NN, inputs):
	a = dict()
	z = dict()
	w, b = ML.separate_weights_and_biases(NN.THETA)
	ndim_ = max(w[1].ndim, inputs.ndim)
	inputs = myNP.normalize_matrix_dimensions(inputs, ndim_)
	if inputs.shape[-1] != w[1].shape[1]: inputs = inputs.T

	a[0] = inputs
	for i in w.keys():
		w[i] = myNP.normalize_matrix_dimensions(w[i], ndim_)
		b[i] = myNP.normalize_matrix_dimensions(b[i], ndim_)
		a[i] = NN.activation_function_formula(a, w, b, i)
	return a
```

---
### backpropagation
Basic formula of backpropagation for updating the matrix THETA, the full implementation and explanation can be found [here](https://www.youtube.com/watch?v=MfIjxPh6Pys&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=11) and [here](https://www.youtube.com/watch?v=zUazLXZZA2U&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=12).

```python
def backpropagation(NN, learning_rate, expected_outputs, a):
	THETA = copy.deepcopy(NN.THETA)
	L = last_layer = max(a.keys())
	h = numpy.ones(a[L].shape) * max(0.001 * numpy.mean(a[L]), 10**-30)

	y = expected_outputs
	dJ_da = (NN.cost_function(y, a[L] + h) - NN.cost_function(y, a[L])) / h
	dz_dw = dz_da = dJ_dw = 0
	dJ_dz = dict()
	w, b = separate_weights_and_biases(THETA)

	for i in range(L, 0, -1):
		da_dz = NN.activation_function_derivate(a, w, b, i)
		dz_dw = a[i-1]

		if i == L: dJ_dz[i] = dJ_da * da_dz
		else:
			dz_da = w[i+1]
			dJ_dz[i] = dJ_dz[i+1] @ dz_da * da_dz

		dJ_dw = dJ_dz[i].T @ dz_dw
		dJ_db = dJ_dz[i].T @ numpy.ones((dJ_dz[i].shape[0],1))
		
		THETA[i] -= learning_rate * numpy.concatenate((dJ_db.T, dJ_dw.T)).T

	return THETA
```

---
### calculate_error
Uses the `cost_function` declared by the user if any, else the default one to calculate the loss for each evaluated input ($\hat{y}$) in respect to the real output ($y$)

```python
def calculate_error(NN, y = None, y_hat = None, cost_function = None):
	if y is None and y_hat is None:
		y = NN.training_set[1]
		y_hat = NN.evaluate(NN.training_set[0])

	if cost_function == None: cost_function = NN.cost_function

	return numpy.mean(cost_function(y, y_hat))
```

---
### train_for_one_epoch
Arguments:
- *(Optional)* `learning_rate` : float
- *(Optional)* `number_of_inputs_per_batch`: int

If we want to run a stochastic approach to the NN we can pass the argument `number_of_inputs_per_batch`  which will divide the inputs in different parts, each with at most a number of values equal to the value of the argument `number_of_inputs_per_batch`.

```python
def train_for_one_epoch(NN, learning_rate = 0.5, number_of_inputs_per_batch = None):
	if number_of_inputs_per_batch is None: number_of_inputs_per_batch = len(NN.training_set[0])
	i = 0
	while (i+1)*number_of_inputs_per_batch <= len(NN.training_set[0]):
		range_of_batch = range(i*number_of_inputs_per_batch, (i+1)*number_of_inputs_per_batch)
		NN.THETA = NN.backpropagation(
			learning_rate = learning_rate,
			expected_outputs = NN.training_set[1][range_of_batch],
			a = NN.feedforward_propagation(NN.training_set[0][range_of_batch])
		)
		i += 1


	if i*number_of_inputs_per_batch < len(NN.training_set[0]):
		range_of_batch = range(i*number_of_inputs_per_batch, len(NN.training_set[0]))
		NN.THETA = NN.backpropagation(
			learning_rate = learning_rate,
			expected_outputs = NN.training_set[1][range_of_batch],
			a = NN.feedforward_propagation(NN.training_set[0][range_of_batch])
		)
```

---
### train_for_n_epochs
Arguments:
- **\[Required\]** `n_epochs` : int
- *(Optional)* `learning_rate` : float
- *(Optional)* `number_of_inputs_per_batch` : int
- *(Optional)* `stopping_criteria` : function
- *(Optional)* `show_plot` : bool
- *(Optional)* `plot_N_times` : int 
	- Determine how many times the function `plot(...)` is called, being a function which takes time.
- *(Optional)* `stop_execution` : bool
	- If true the execution is stopped until the plot window is closed

```python
def no_stopping_criteria(J_validation_history, J_training_history):
	return False


def train_for_n_epochs(
	NN,
	n_epochs, 
	learning_rate = 0.5, 
	number_of_inputs_per_batch = None,
	*,
	stopping_criteria = no_stopping_criteria,
	show_plot = True,
	plot_N_times = 100,
	stop_execution = True,
):
	Plot = new_plot4()
	Plot.legend[0] = "TRAINING_ERROR"
	Plot.legend[1] = "VALIDATION_ERROR"
	if show_plot == True: Plot.show()
	buffer_error_trn = tuple()
	buffer_error_val = tuple()
	J_training_history = tuple()
	J_validation_history = tuple()
	y_training = NN.training_set[1]
	y_validation = NN.validation_set[1]
	y_hat_training = NN.evaluate(NN.training_set[0])
	y_hat_validation = NN.evaluate(NN.validation_set[0])
	min_validation_error = NN.calculate_error(y = y_validation, y_hat = y_hat_validation)
	best_THETA = copy.deepcopy(NN.THETA)

	for i in range(n_epochs+1):
		NN.train_for_one_epoch(learning_rate, number_of_inputs_per_batch)
		y_hat_training = NN.evaluate(NN.training_set[0])
		y_hat_validation = NN.evaluate(NN.validation_set[0])
		training_error = NN.calculate_error(y = y_training, y_hat = y_hat_training)
		validation_error = NN.calculate_error(y = y_validation, y_hat = y_hat_validation)
		if validation_error < min_validation_error: best_THETA = copy.deepcopy(NN.THETA)

		if show_plot == True and (n_epochs <= plot_N_times or (i % round(n_epochs/plot_N_times) == 0)):
			Plot.add_data(buffer_error_trn, buffer_error_val)
			buffer_error_trn = (training_error,)
			buffer_error_val = (validation_error,)
		else:
		 	buffer_error_trn += (training_error,)
		 	buffer_error_val += (validation_error,)

		J_training_history += (training_error,)
		J_validation_history += (validation_error,)
		if stopping_criteria(J_training_history, J_validation_history) == True: break

	if show_plot == True and stop_execution == True: Plot.end()
	NN.THETA = copy.deepcopy(best_THETA)
	return best_THETA
```

---
### evaluate
Returns the last layer output of the NN.
```python
def evaluate(NN, inputs):
	a = NN.feedforward_propagation(inputs)
	return a[max(a.keys())]
```

---
### get_numpy_matrices_from_MNIST_set
```python
def get_numpy_matrices_from_MNIST_set(file_path):
	with gzip.open(file_path, 'r') as f:
		# first 4 bytes is a magic number
		magic_number = int.from_bytes(f.read(4), 'big')
		# second 4 bytes is the number of images
		image_count = int.from_bytes(f.read(4), 'big')
		# third 4 bytes is the row count
		row_count = int.from_bytes(f.read(4), 'big')
		# fourth 4 bytes is the column count
		column_count = int.from_bytes(f.read(4), 'big')
		# rest is the image pixel data, each pixel is stored as an unsigned byte
		# pixel values are 0 to 255
		image_data = f.read()
		images = numpy.frombuffer(image_data, dtype=numpy.uint8)\
			.reshape((image_count, row_count, column_count))
		return images
```

### get_numpy_array_from_MNIST_labels
```python
def get_numpy_array_from_MNIST_labels(file_path):
	with gzip.open(f"{FOLDER_NAME}/{TRAINING_LABELS_NAME}", 'r') as f:
		# first 4 bytes is a magic number
		magic_number = int.from_bytes(f.read(4), 'big')
		# second 4 bytes is the number of labels
		label_count = int.from_bytes(f.read(4), 'big')
		# rest is the label data, each label is stored as unsigned byte
		# label values are 0 to 9
		label_data = f.read()
		labels = numpy.frombuffer(label_data, dtype=numpy.uint8)
		return labels
```

### create_NN
```python
def create_NN(cost_function, activation_function):
	inputs = numpy.array(((0,0),(0,0)))
	outputs = numpy.array(((0,0),(0,0)))
	return mp.Classes.General_Backpropagation_NN(
		inputs, outputs,
		cost_function = cost_function,
		activation_function = activation_function,
	)
```

### Dictionaries for one hot bit encoding
```python
def create_classification_dict(training_labels):
		training_labels = convert_numpy_array_to_tuple(training_labels)
		value_to_class = dict()
		class_to_value = dict()
		index = 0
		for value in training_labels:
			if value not in value_to_class.keys():
				index += 1
				value_to_class[value] = index
				class_to_value[index] = value
		return value_to_class, class_to_value
```

### classify_labels
```python
def classify_label(labels_array, value_to_class_dictionary):
		assert type(labels_array) is numpy.ndarray
		lables_tuple = convert_numpy_array_to_tuple(labels_array)
		class_array = numpy.zeros((labels_array.shape[0], max(class_to_value.keys()) + 1))
		for i in range(labels_array.shape[0]):
			value_ = lables_tuple[i]
			class_ = value_to_class_dictionary[value_]
			class_array[i][class_] = 1
		return class_array
```