/**
 * A mini library for Artificial Neural Network inspired by ToyNeuralNetwork - https://github.com/CodingTrain/Toy-Neural-Network-JS
 */
class NeuralNetwork {
  /**
   * @param  {Array} arg_array - array of counts of neurons in each layer
   * Eg : new NeuralNetwork([3,4,2]); will instantiate ANN with 3 neurons as input layer, 4 as hidden and 2 as output layer
   */
  constructor(arg_array) {

    this.layer_nodes_counts = arg_array; // no of neurons per layer
    this.layers_count = arg_array.length; //total number of layers

    this.weights = []; //array of weights matrices in order

    const {layer_nodes_counts} = this;

    for(let i = 0 ; i < layer_nodes_counts.length - 1 ; i++) {
      let weights_mat = new Matrix(layer_nodes_counts[i+1],layer_nodes_counts[i]);
      weights_mat.randomize()
      this.weights.push(weights_mat);
    }

    this.biases = []; //array of bias matrices in order

    for(let i = 1 ; i < layer_nodes_counts.length ; i++) {
      let bias_mat = new Matrix(layer_nodes_counts[i],1);
      bias_mat.randomize()
      this.biases.push(bias_mat);
    }

    NeuralNetwork.SIGMOID = 1;
    NeuralNetwork.ReLU = 2;

    this.activation = null;
    this.activation_derivative = null;
    this.setActivation(NeuralNetwork.SIGMOID);
    this.learningRate = 0.2;
  }

  /**
   *
   * @param {Array} input_array - Array of input values
   * @param {Boolean} GET_ALL_LAYERS - if we need all layers after feed forward instead of just output layer
   */
  feedforward(input_array, GET_ALL_LAYERS) {
    const {layers_count} = this;

    if(!this._feedforward_args_validator(input_array)) {
      return -1;
    }

    let layers = []; //This will be array of layer arrays

    //input layer
    layers[0] = Matrix.fromArray(input_array);

    for(let i = 1 ; i < layers_count ; i++) {
      layers[i] = Matrix.multiply(this.weights[i-1],layers[i-1]);
      layers[i].add(this.biases[i-1]);
      layers[i].map(this.activation); //activation
    }
    if (GET_ALL_LAYERS == true) {
      return layers; //all layers (array of layer matrices)
    } else {
      return layers[layers.length-1].toArray(); //output layer array
    }
  }

  // Mutates weights and biases of ANN based on rate given
  mutate(rate) { //rate 0 to 1
    function mutator(val) {
      if(Math.random() < rate) {
        return val + Math.random() * 2 - 1;
      }
      else {
        return val;
      }
    }

    for(let i=0 ; i < this.weights.length ; i++) {
      this.weights[i].map(mutator);
      this.biases[i].map(mutator);
    }
  }

  // Returns a copy of ANN (instead of reference to original one)
  copy() {
    let new_ann = new NeuralNetwork(this.layer_nodes_counts);
    for(let i=0 ; i< new_ann.weights.length ; i++) {
      new_ann.weights[i] = this.weights[i].copy();
    }
    for(let i=0 ; i< new_ann.biases.length ; i++) {
      new_ann.biases[i] = this.biases[i].copy();
    }
    return new_ann;
  }

  // Trains with backpropogation
  train(input_array, target_array) {
    if(!this._train_args_validator(input_array, target_array)) {
      return -1;
    }

    let layers = this.feedforward(input_array, true); //layer matrices
    let target_matrix = Matrix.fromArray(target_array);

    let prev_error;

    for(let layer_index = layers.length-1; layer_index >= 1; layer_index--) {
      /* right and left are in respect to the current layer */
      let layer_matrix = layers[layer_index];

      let layer_error;
      //Error calculation
      if(layer_index == layers.length-1) { // Output layer
        layer_error = Matrix.add(target_matrix, Matrix.multiply(layer_matrix, -1));
      } else { //Hidden layer
        const right_weights = this.weights[layer_index];
        const right_weigths_t = Matrix.transpose(right_weights);
        layer_error = Matrix.multiply(right_weigths_t, prev_error);
      }
      prev_error = layer_error.copy(); //will be used for error calculation in hidden layers

      //Calculating layer gradient
      const layer_gradient = Matrix.map(layer_matrix, this.activation_derivative);
      layer_gradient.multiply(layer_error);
      layer_gradient.multiply(this.learningRate);

      //Calculating delta weights
      const left_layer_t = Matrix.transpose(layers[layer_index-1]);
      const left_weights_delta = Matrix.multiply(layer_gradient, left_layer_t);

      //Updating weights and biases
      this.weights[layer_index-1].add(left_weights_delta);
      this.biases[layer_index-1].add(layer_gradient);
    }
  }

  activation(x) {
    return this.activation(x);
  }

  setActivation(TYPE) {
    switch (TYPE) {
      case NeuralNetwork.SIGMOID:
        this.activation = NeuralNetwork.sigmoid;
        this.activation_derivative = NeuralNetwork.sigmoid_derivative;
        break;
      case NeuralNetwork.ReLU:
        this.activation = NeuralNetwork.relu;
        this.activation_derivative = NeuralNetwork.relu_derivative;
        break;
      default:
        console.error('Activation type invalid, setting sigmoid by default');
        this.activation = NeuralNetwork.sigmoid;
        this.activation_derivative = NeuralNetwork.sigmoid_derivative;
    }
  }

  /**
   * @param {NeuralNetwork} ann - crossover partner
   */
  crossover(ann) {
    if(!this._crossover_validator(ann)) {
      return -1;
    }
    const offspring = new NeuralNetwork(this.layer_nodes_counts);
    for(let i = 0; i < this.weights.length; i++) {
      if(Math.random() < 0.5)  {
        offspring.weights[i] = this.weights[i];
      } else {
        offspring.weights[i] = ann.weights[i];
      }

      if(Math.random() < 0.5)  {
        offspring.biases[i] = this.biases[i];
      } else {
        offspring.biases[i] = ann.biases[i];
      }
    }
    return offspring;
  }

  // Activation functions
  static sigmoid(x) {
    return 1 / ( 1 + Math.exp(-1 * x));
  }

  static sigmoid_derivative(y) {
    return y*(1-y);
  }

  static relu(x) {
    if(x >= 0) {
      return x;
    } else {
      return 0;
    }
  }

  static relu_derivative(y) {
    if(y > 0) {
      return 1;
    } else {
      return 0;
    }
  }

  // Argument validator functions
  _feedforward_args_validator(input_array) {
    let invalid = false;
    if(input_array.length != this.layer_nodes_counts[0]) {
      invalid = true;
      console.error("Feedforward failed : Input array and input layer size doesn't match.");
    }
    return invalid ? false : true;
  }

  _train_args_validator(input_array, target_array) {
    let invalid = false;
    if(input_array.length != this.layer_nodes_counts[0]) {
      console.error("Training failed : Input array and input layer size doesn't match.");
      invalid=true;
    }
    if(target_array.length != this.layer_nodes_counts[this.layers_count - 1]) {
      invalid=true;
      console.error("Training failed : Target array and output layer size doesn't match.");
    }
    return invalid ? false : true;
  }

  _crossover_validator(ann) {
    let invalid = false;
    if(ann instanceof NeuralNetwork) {
      if(this.layers_count == ann.layers_count) {
        for(let i = 0; i < this.layers_count; i++) {
          if(this.layer_nodes_counts[i] != ann.layer_nodes_counts[i]) {
            console.error("Crossover failed : Architecture mismatch (Different number of neurons in one or more layers).");
            invalid = true;
            break;
          }
        }
      } else {
        invalid=true;
        console.error("Crossover failed : Architecture mismatch (Different number of layers).");
      }
    } else {
      invalid=true;
      console.error("Crossover failed : NeuralNetwork object expected.");
    }
    return invalid ? false : true;
  }
}
