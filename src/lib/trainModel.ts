import * as tf from '@tensorflow/tfjs';

// Enhanced feature extraction from raw accelerometer data
function extractMotionFeatures(data: number[][]): number[][] {
  const features: number[][] = [];
  const windowSize = 10; // Look at 10 consecutive readings
  
  for (let i = windowSize; i < data.length; i++) {
    const window = data.slice(i - windowSize, i);
    const feature = calculateMotionFeatures(window);
    features.push(feature);
  }
  
  return features;
}

export function calculateMotionFeatures(window: number[][]): number[] {
  const features: number[] = [];
  
  // Extract x, y, z arrays from window
  const x = window.map(d => d[0]);
  const y = window.map(d => d[1]);
  const z = window.map(d => d[2]);
  
  // 1. Raw values (current reading)
  const current = window[window.length - 1];
  features.push(...current); // [x, y, z]
  
  // 2. Statistical features for each axis
  for (const axis of [x, y, z]) {
    features.push(
      mean(axis),                    // Average value
      variance(axis),                // How spread out values are
      Math.max(...axis) - Math.min(...axis), // Range of movement
    );
  }
  
  // 3. Rate of change (velocity approximation)
  const xVelocity = x.map((val, i) => i > 0 ? val - x[i-1] : 0);
  const yVelocity = y.map((val, i) => i > 0 ? val - y[i-1] : 0);
  const zVelocity = z.map((val, i) => i > 0 ? val - z[i-1] : 0);
  
  features.push(
    mean(xVelocity.map(Math.abs)),  // Average speed of change in X
    mean(yVelocity.map(Math.abs)),  // Average speed of change in Y
    mean(zVelocity.map(Math.abs)),  // Average speed of change in Z
  );
  
  // 4. Acceleration (rate of change of velocity)
  const xAccel = xVelocity.map((val, i) => i > 0 ? val - xVelocity[i-1] : 0);
  const yAccel = yVelocity.map((val, i) => i > 0 ? val - yVelocity[i-1] : 0);
  const zAccel = zVelocity.map((val, i) => i > 0 ? val - zVelocity[i-1] : 0);
  
  features.push(
    mean(xAccel.map(Math.abs)),     // Jerk in X direction
    mean(yAccel.map(Math.abs)),     // Jerk in Y direction
    mean(zAccel.map(Math.abs)),     // Jerk in Z direction
  );
  
  // 5. Dominant axis analysis
  const xActivity = variance(x);
  const yActivity = variance(y);
  const zActivity = variance(z);
  const totalActivity = xActivity + yActivity + zActivity;
  
  if (totalActivity > 0) {
    features.push(
      xActivity / totalActivity,    // X dominance ratio
      yActivity / totalActivity,    // Y dominance ratio
      zActivity / totalActivity,    // Z dominance ratio
    );
  } else {
    features.push(0.33, 0.33, 0.33); // Equal if no movement
  }
  
  // 6. Cross-axis correlation (circular motion detection)
  features.push(
    correlation(x, y),              // X-Y correlation
    correlation(y, z),              // Y-Z correlation
    correlation(x, z),              // X-Z correlation
  );
  
  // 7. Movement intensity
  const magnitude = window.map(d => Math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]));
  features.push(
    mean(magnitude),                // Average magnitude
    variance(magnitude),            // Magnitude variation
  );
  
  return features;
}

// Helper functions
function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function variance(arr: number[]): number {
  const m = mean(arr);
  return mean(arr.map(x => (x - m) * (x - m)));
}

function correlation(x: number[], y: number[]): number {
  const n = x.length;
  const meanX = mean(x);
  const meanY = mean(y);
  
  let numerator = 0;
  let sumXSquared = 0;
  let sumYSquared = 0;
  
  for (let i = 0; i < n; i++) {
    const deltaX = x[i] - meanX;
    const deltaY = y[i] - meanY;
    numerator += deltaX * deltaY;
    sumXSquared += deltaX * deltaX;
    sumYSquared += deltaY * deltaY;
  }
  
  const denominator = Math.sqrt(sumXSquared * sumYSquared);
  return denominator === 0 ? 0 : numerator / denominator;
}

// Enhanced CSV processing with temporal features
function processCSV(csvText: string): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',');
  const dataLines = lines.slice(1);

  // Group data by class for temporal processing
  const classData: { [key: number]: number[][] } = {};
  const classLabels: { [key: number]: number[][] } = {};

  for (const line of dataLines) {
    const [classStr, x, y, z] = line.split(',').map(Number);
    
    if (!classData[classStr]) {
      classData[classStr] = [];
      classLabels[classStr] = [];
    }
    
    classData[classStr].push([x, y, z]);
    
    // One-hot encoding
    if (classStr === 0) classLabels[classStr].push([1, 0, 0, 0]);      // horizontal
    else if (classStr === 1) classLabels[classStr].push([0, 1, 0, 0]); // vertical
    else if (classStr === 2) classLabels[classStr].push([0, 0, 1, 0]); // still
    else classLabels[classStr].push([0, 0, 0, 1]);                    // circular
  }

  const allFeatures: number[][] = [];
  const allLabels: number[][] = [];

  // Extract features for each class
  for (const classNum in classData) {
    const rawData = classData[classNum];
    const labels = classLabels[classNum];
    
    if (rawData.length < 10) continue; // Need at least 10 samples for window
    
    const features = extractMotionFeatures(rawData);
    
    // Match features with labels (skip first 10 labels due to windowing)
    for (let i = 0; i < features.length; i++) {
      allFeatures.push(features[i]);
      allLabels.push(labels[i + 10]); // Offset by window size
    }
  }

  console.log(`Extracted ${allFeatures.length} feature vectors with ${allFeatures[0]?.length || 0} features each`);

  return {
    xs: tf.tensor2d(allFeatures),
    ys: tf.tensor2d(allLabels)
  };
}

export async function createAndTrainModelFromCSV(csvText: string, onEpochEnd?: (epoch: number, logs: any) => void) {
  const { xs, ys } = processCSV(csvText);

  // Enhanced model architecture for feature-rich input
  const model = tf.sequential();
  
  // Input layer sized for rich features (≈23 features)
  model.add(tf.layers.dense({ 
    inputShape: [xs.shape[1]], 
    units: 64, 
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  }));
  
  model.add(tf.layers.dropout({ rate: 0.3 }));
  
  model.add(tf.layers.dense({ 
    units: 32, 
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  }));
  
  model.add(tf.layers.dropout({ rate: 0.3 }));
  
  model.add(tf.layers.dense({ 
    units: 16, 
    activation: 'relu' 
  }));
  
  // Output layer for 4 classes
  model.add(tf.layers.dense({ 
    units: 4, 
    activation: 'softmax' 
  }));

  model.compile({
    optimizer: tf.train.adam(0.001), // Lower learning rate for stability
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  console.log('Model summary:');
  model.summary();

  await model.fit(xs, ys, {
    epochs: 30,
    validationSplit: 0.2, // Use 20% for validation
    shuffle: true,
    batchSize: 32,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs?.loss?.toFixed(4)}, accuracy = ${logs?.acc?.toFixed(4)}, val_loss = ${logs?.val_loss?.toFixed(4)}, val_acc = ${logs?.val_acc?.toFixed(4)}`);
        if (onEpochEnd) onEpochEnd(epoch, logs);
      }
    }
  });

  await model.save('indexeddb://motion-model');
  
  // Clean up tensors
  xs.dispose();
  ys.dispose();
  
  return model;
}
