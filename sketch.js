// DATASET - POINTS
const x_values = [];
const y_values = [];

let m, b;

const _learningRate = 0.5;
const _optmizer = tf.train.sgd(_learningRate);

function setup() {
  createCanvas(400, 400);

  // initialize the m and b weight
  // the data set is immutable, so the line has to update its m and b values to fit the line
  // for that we use tf variable to be able to change the tensor
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

// predictions is the y values return from predict
// labels are the real y from the dataset
function loss(predictions, labels) {
  return predictions.sub(labels).square().mean();
}

// receives the x values and predict and y output
// the output will be used to calculate the loss
function predict(xs) {
  //convert xs to a tensor
  const _tensor_xs = tf.tensor1d(xs);

  //formula to line  - y = mx + b;
  const ys = _tensor_xs.mul(m).add(b);

  return ys;
}

function mousePressed() {
  //For each mouse click, map the mouse position to a normalization between 0 and 1
  const x = map(mouseX, 0, width, 0, 1);
  const y = map(mouseY, 0, height, 1, 0);
  x_values.push(x);
  y_values.push(y);
}

function draw() {
  // Training the model
  // Receives two tensors
  tf.tidy(() => {
    x_values.length > 0
      ? _optmizer.minimize(() => loss(predict(x_values), tf.tensor1d(y_values)))
      : null;
  });

  background(0);
  stroke(255);
  strokeWeight(8);

  for (let i = 0; i < x_values.length; i++) {
    // Denormalize the point stored
    let px = map(x_values[i], 0, 1, 0, width);
    let py = map(y_values[i], 0, 1, height, 0);
    point(px, py);
  }

  let xs = [0, 1];
  const ys = tf.tidy(() => predict(xs));
  let lineY = ys.dataSync();
  ys.dispose();

  let x1 = map(xs[0], 0, 1, 0, width);
  let x2 = map(xs[1], 0, 1, 0, width);

  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);
  strokeWeight(2);
  x_values.length > 0 ? line(x1, y1, x2, y2) : null;
}
