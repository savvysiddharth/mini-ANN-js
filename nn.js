/**
 * A mini library for Artificial Neural Network inspired by ToyNeuralNetwork - https://github.com/CodingTrain/Toy-Neural-Network-JS
 */
class NeuralNetwork {
  /**
   * @param  {Array} layers_nodes_count - integer counts of neurons in each layer
   * Eg : new NeuralNetwork([3,4,2]); will instantiate ANN with 3 neurons as input layer, 4 as hidden and 2 as output layer
   */
  constructor(arg_array) {

    this.layers_nodes_count = arg_array; // no of neurons per layer

    this.weights = []; //array of weights matrices in order

    const {layers_nodes_count} = this;

    for(let i = 0 ; i < layers_nodes_count.length - 1 ; i++) {
      let weights_mat = new Matrix(layers_nodes_count[i+1],layers_nodes_count[i]);
      weights_mat.randomize()
      this.weights.push(weights_mat);
    }

    this.biases = []; //array of bias matrices in order

    for(let i = 1 ; i < layers_nodes_count.length ; i++) {
      let bias_mat = new Matrix(layers_nodes_count[i],1);
      bias_mat.randomize()
      this.biases.push(bias_mat);
    }
  }

  feedforward(input_array) {
    const {layers_nodes_count} = this;
    //argument validation code - START
    let invalid=false;
    if(input_array.length != layers_nodes_count[0]) {
      invalid=true;
      console.error("Input array to feedforward function has invalid size.");
    }
    if(invalid) {
      console.error("Network feedforward failed : invalid arguments!");
      return -1;
    }
    //argument validation code - END

    let layers = []; //In this array actual (1d)matrix of neurons in each layer will be stored

    //input layer
    layers[0] = Matrix.fromArray(input_array);

    for(let i = 1 ; i < layers_nodes_count.length ; i++) {
      layers[i] = Matrix.multiply(this.weights[i-1],layers[i-1]);
      layers[i].add(this.biases[i-1]);
      layers[i].map(sigmoid); //activation
    }
    return layers[layers.length-1].toArray(); //output layer array
  }

  mutate(rate) { //rate 0 to 1
    function mutate(val) {
      if(Math.random() < rate) {
        return val + Math.random() * 2 - 1;
        // return val + randomGaussian(0,0.1);
      }
      else {
        return val;
      }
    }

    for(let i=0 ; i < this.weights.length ; i++) {
      this.weights[i].map(mutate);
      this.biases[i].map(mutate);
    }
  }

  copy() {

    let newann = new NeuralNetwork(this.layers_nodes_count);

    for(let i=0 ; i< newann.weights.length ; i++) {
      newann.weights[i] = this.weights[i].copy();
    }

    for(let i=0 ; i< newann.biases.length ; i++) {
      newann.biases[i] = this.biases[i].copy();
    }

    return newann;
  }
}

function sigmoid(x)
{
  return 1/(1+Math.exp(-1*x));
}