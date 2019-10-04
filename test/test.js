function AND_gate_test() {
  console.log("AND GATE TEST : ");

  let mynn = new NeuralNetwork([2,1]);

  let training_data = [
    [[1,1], [1]],
    [[1,0], [0]],
    [[0,1], [0]],
    [[0,0], [0]],
  ];

  console.log("Before training...");
  console.log('0 AND 0',mynn.feedforward([0,0]));
  console.log('0 AND 1',mynn.feedforward([0,1]));
  console.log('1 AND 0',mynn.feedforward([1,0]));
  console.log('1 AND 1',mynn.feedforward([1,1]));

  // several training epochs
  for(let i=0; i<10000; i++) {
    let tdata = training_data[Math.floor(Math.random() * training_data.length)];
    mynn.train(tdata[0], tdata[1]);
  }

  console.log("After training...");
  console.log('0 AND 0',mynn.feedforward([0,0]));
  console.log('0 AND 1',mynn.feedforward([0,1]));
  console.log('1 AND 0',mynn.feedforward([1,0]));
  console.log('1 AND 1',mynn.feedforward([1,1]));
}

AND_gate_test();