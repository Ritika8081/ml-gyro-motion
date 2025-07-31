import * as tf from '@tensorflow/tfjs';

// Utility to convert CSV text to tensors
function processCSV(csvText: string): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(','); // ['class', 'horizontal', 'vertical', 'still']
  const dataLines = lines.slice(1);

  const xs: number[][] = [];
  const ys: number[][] = [];

  for (const line of dataLines) {
    const [classStr, x, y, z] = line.split(',').map(Number);
    xs.push([x, y, z]);
    if (classStr === 0) ys.push([1, 0, 0]);       // horizontal
    else if (classStr === 1) ys.push([0, 1, 0]);  // vertical
    else ys.push([0, 0, 1]);                      // still
  }

  return {
    xs: tf.tensor2d(xs),
    ys: tf.tensor2d(ys)
  };
}

export async function createAndTrainModelFromCSV(csvText: string) {
  const { xs, ys } = processCSV(csvText);

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [3], units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  await model.fit(xs, ys, {
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (logs) {
          console.log(`Epoch ${epoch + 1}: loss=${logs.loss?.toFixed(4)}, accuracy=${(logs.acc ? (logs.acc * 100).toFixed(2) : 'N/A')}%`);
        }
      }
    }
  });

  await model.save('indexeddb://motion-model');
  return model;
}
