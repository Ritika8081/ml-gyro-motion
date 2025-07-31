import * as tf from '@tensorflow/tfjs';

export async function createAndTrainModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [3], units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // 3 output classes

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Sample training data
  const xs = tf.tensor2d([
    [0.01, 9.8, 0.02], // vertical
    [9.6, 0.03, 0.1],  // horizontal
    [0.05, 0.02, 0.01] // still
    // add more real data here
  ]);

  const ys = tf.tensor2d([
    [0, 1, 0], // vertical
    [1, 0, 0], // horizontal
    [0, 0, 1], // still
  ]);

  await model.fit(xs, ys, {
    epochs: 100,
    shuffle: true,
  });

  await model.save('indexeddb://motion-model');
  return model;
}
