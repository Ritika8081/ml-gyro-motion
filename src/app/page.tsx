'use client'
import { useEffect, useState } from 'react';
import { createAndTrainModel } from '../lib/trainModel';
import * as tf from '@tensorflow/tfjs';

type SerialPort = any; // Add this line to declare SerialPort type

export default function Home() {
const [reader, setReader] = useState<ReadableStreamDefaultReader | null>(null);
  const [port, setPort] = useState<SerialPort | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [a, setA] = useState(0); // ax
  const [b, setB] = useState(0); // ay
  const [c, setC] = useState(0); // az
  const [motion, setMotion] = useState<string | null>(null);
  const [history, setHistory] = useState<{ a: number, b: number, c: number }[]>([]);
  const labels = ['horizontal', 'vertical', 'still'];

  useEffect(() => {
    const loadOrTrainModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('indexeddb://motion-model');
        loadedModel.compile({
          optimizer: tf.train.adam(0.01),
          loss: 'categoricalCrossentropy',
        });
        setModel(loadedModel);
        console.log('✅ Motion model loaded and compiled');
      } catch (error) {
        console.log('⚠️ No motion model found. Training a new one...');
        const trainedModel = await createAndTrainModel();
        setModel(trainedModel);
      }
    };

    loadOrTrainModel();
  }, []);

  useEffect(() => {
  if (!reader || !model) return;

  let cancelled = false;

  const readLoop = async () => {
    while (!cancelled) {
      try {
        const { value, done } = await reader.read();
        if (done || !value) break;

        console.log('Serial value:', value); // <-- Add this line

        const [ax, ay, az] = value.trim().split(',').map(Number);
        if ([ax, ay, az].some(v => isNaN(v))) {
          console.warn('Invalid serial input:', value);
          continue; // Skip this reading, but keep reading
        }

        setA(ax);
        setB(ay);
        setC(az);

        setHistory(prev => [
          { a: ax, b: ay, c: az },
          ...prev.slice(0, 19)
        ]);

        if (model) {
          const input = tf.tensor2d([[ax, ay, az]]);
          const output = model.predict(input) as tf.Tensor;
          const result = await output.data();
          const predictedIndex = result.indexOf(Math.max(...result));
          setMotion(labels[predictedIndex]);
        }
      } catch (err) {
        console.warn('Read loop error:', err);
      }
    }
  };

  readLoop();

  return () => { cancelled = true; };
}, [reader, model]);


  const handlePredict = async () => {
    if (!model) return;
    const input = tf.tensor2d([[a, b, c]]);
    const prediction = model.predict(input) as tf.Tensor;
    const result = await prediction.data();
    const predictedIndex = result.indexOf(Math.max(...result));
    setMotion(labels[predictedIndex]);
  };
  
  useEffect(() => {
  return () => {
    reader?.cancel();
    port?.close();
  };
}, [reader, port]);


  return (
    <main className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-2 sm:p-4">
      <div className="w-full max-w-xl">
        <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-center">
          Motion Classifier using TensorFlow.js 🧠
        </h1>
        <button
          onClick={async () => {
            const newPort = await (navigator as any).serial.requestPort();
            await newPort.open({ baudRate: 230400 });
            const textDecoder = new TextDecoderStream();
            const readableStreamClosed = newPort.readable?.pipeTo(textDecoder.writable);
            const newReader = textDecoder.readable.getReader();
            setPort(newPort);
            setReader(newReader);
            console.log('✅ Serial connected');
          }}
          className="w-full sm:w-auto bg-purple-600 text-white px-4 py-2 rounded mb-4 shadow hover:bg-purple-700 transition"
        >
          Connect to Accelerometer
        </button>

        {/* Live Accelerometer Data */}
        <div className="mb-4 p-4 bg-white rounded shadow flex flex-col sm:flex-row justify-between items-center gap-2">
          <div className="text-base sm:text-lg">
            <span className="font-semibold">X axis (ax): </span>
            <span className="font-mono">{a}</span>
          </div>
          <div className="text-base sm:text-lg">
            <span className="font-semibold">Y axis (ay): </span>
            <span className="font-mono">{b}</span>
          </div>
          <div className="text-base sm:text-lg">
            <span className="font-semibold">Z axis (az): </span>
            <span className="font-mono">{c}</span>
          </div>
        </div>

        {/* Manual Input & Predict */}
        <div className="flex flex-col sm:flex-row gap-2 mb-4">
          <input
            type="number"
            className="border p-2 rounded flex-1"
            placeholder="X axis (ax)"
            value={a}
            onChange={(e) => setA(parseFloat(e.target.value))}
          />
          <input
            type="number"
            className="border p-2 rounded flex-1"
            placeholder="Y axis (ay)"
            value={b}
            onChange={(e) => setB(parseFloat(e.target.value))}
          />
          <input
            type="number"
            className="border p-2 rounded flex-1"
            placeholder="Z axis (az)"
            value={c}
            onChange={(e) => setC(parseFloat(e.target.value))}
          />
          <button
            className="bg-blue-600 text-white px-4 py-2 rounded flex-1 sm:flex-none"
            onClick={handlePredict}
          >
            Predict Motion
          </button>
        </div>

        {/* Prediction Result */}
        {motion && (
          <p className="text-lg sm:text-xl font-medium text-center mb-4">
            Predicted Motion: <span className="font-bold">{motion}</span>
          </p>
        )}

        {/* Recent Data Stream */}
        <div className="my-4 p-4 bg-white rounded shadow text-sm max-h-64 overflow-y-auto w-full">
          <div className="font-bold mb-2 text-center">Recent Data Stream:</div>
          {history.length === 0 && <div className="text-gray-400 text-center">No data yet.</div>}
          {history.map((item, idx) => (
            <div key={idx} className="flex justify-between gap-2 border-b last:border-b-0 py-1">
              <span>X: <span className="font-mono">{item.a}</span></span>
              <span>Y: <span className="font-mono">{item.b}</span></span>
              <span>Z: <span className="font-mono">{item.c}</span></span>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
