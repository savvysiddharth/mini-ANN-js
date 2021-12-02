![mini-ann](https://user-images.githubusercontent.com/12862695/65754632-15627f80-e12f-11e9-8333-5fa8112a6af1.png)

## A lightweight Neural Network library in Javascript

This library is inspired by [Toy-Neural-Network](https://github.com/CodingTrain/Toy-Neural-Network-JS), which works for one hidden layer. *mini-ANN-js* provides basic ANN functionalities which includes ability to create multilayer architecture, feed forward, training through backpropagation and some functions for genetic algorithm.

### Documentation

* **Initialize** Neural Network

```javascript
// ANN with 4 inputs, 3 neurons in hidden layer and 2 outputs
const my_ann = new NeuralNetwork([4, 3, 2]);
// initializes ANN with random weights and biases
```

* Changing **Activation function**

```javascript
// set ReLU function as activation function
my_ann.setActivation(NeuralNetwork.ReLU);

// set Sigmoid function as activation function
my_ann.setActivation(NeuralNetwork.SIGMOID);

// By default Sigmoid is the activation function
```

* Performing **feed forward**

```javascript
// passing 4 inputs as follows...
const output = my_ann.feedforward([0, 2, 1, 2]);
// returns an array with outputs of ANN

// pass 2nd arg for feedforward as true to get all layers instead of just output
const all_layers = my_ann.feedforward([0, 2, 1, 2], true);

```

* **Training** ANN

```javascript
let input = [0, 2, 1, 2];
let expected_output = [1,0];
my_ann.train(input, expected_output);
```

* Functions for **Genetic algorithms**

```javascript
// mutate weights and biases of ANN
my_ann.mutate(0.2); //mutation rate = 0.2 (min-0 & max-1)

// creates a copy of ann
new_ann = my_ann.copy()

// crossover
const offspring = my_ann.crossover(other_ann);
```

* **Visualization** function

```javascript
const canvas = document.getElementById("canvas")
const ctx = canvas.getContext("2d")
const nn = new NeuralNetwork([3,5,3,2])
nn.draw(
  ctx, // canvas context
  100, // top left x coordinate to start drawing from
  100, // top left y coordinate to start drawing from
  [-0.5, 0.75, 0.2], // OPTIONAL network inputs. When given also visualizes activation states
  20 // OPTIONAL neuron radius. Default = 20
)
```

Example output:

![example output of draw function](https://user-images.githubusercontent.com/543577/144102996-ce1a7e90-ed12-4134-af05-70d46db34ce0.png)