'use client'
import { useEffect, useState, useRef } from 'react';
import { createAndTrainModelFromCSV } from '../lib/trainModel';
import * as tf from '@tensorflow/tfjs';
import { Activity, Zap, Target, TrendingUp, Wifi, WifiOff } from 'lucide-react';

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
  const [recording, setRecording] = useState(false);
  const [selectedClass, setSelectedClass] = useState<number | null>(null);
  const [recordedData, setRecordedData] = useState<{ classLabel: number, ax: number, ay: number, az: number }[]>([]);
  const [dummy, setDummy] = useState(0); // for re-render
  const recordingRef = useRef(false);
  const selectedClassRef = useRef<number | null>(null);
  const labels = ['Class 0', 'Class 1', 'Class 2'];

  useEffect(() => {
    setModel(null); // No model loaded by default, must upload CSV
  }, []);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('indexeddb://motion-model');
        setModel(loadedModel);
        console.log('✅ Model loaded from local storage');
      } catch (err) {
        setModel(null);
        console.log('ℹ️ No saved model found. Please upload a CSV to train.');
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    if (!reader || !model) return;

    let cancelled = false;

    const readLoop = async () => {
      while (!cancelled) {
        try {
          const { value, done } = await reader.read();
          if (done || !value) break;

          const [ax, ay, az] = value.trim().split(',').map(Number);
          if ([ax, ay, az].some(v => isNaN(v))) {

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

          if (recordingRef.current && selectedClassRef.current !== null) {
            setRecordedData(prev => [
              ...prev,
              {
                classLabel: selectedClassRef.current!,
                ax,
                ay,
                az
              }
            ]);
          }

        } catch (err) {
          console.warn('Read loop error:', err);
        }
      }
    };

    readLoop();

    return () => { cancelled = true; };
  }, [reader, model]);

  const getMotionIcon = (motion: string) => {
    switch (motion) {
      case 'Class 0': return <TrendingUp className="w-6 h-6" />;
      case 'Class 1': return <Activity className="w-6 h-6" />;
      case 'Class 2': return <Target className="w-6 h-6" />;
      default: return <Zap className="w-6 h-6" />;
    }
  };

  const getMotionColor = (motion: string) => {
    switch (motion) {
      case 'Class 0': return 'text-blue-600 bg-blue-50';
      case 'Class 1': return 'text-blue-700 bg-blue-100';
      case 'Class 2': return 'text-blue-800 bg-blue-200';
      default: return 'text-blue-600 bg-blue-50';
    }
  };

  const handlePredict = async () => {
    if (!model) return;
    const input = tf.tensor2d([[a, b, c]]);
    const prediction = model.predict(input) as tf.Tensor;
    const result = await prediction.data();
    const predictedIndex = result.indexOf(Math.max(...result));
    setMotion(labels[predictedIndex]);
  };

  useEffect(() => {
    selectedClassRef.current = selectedClass;
  }, [selectedClass]);

  useEffect(() => {
    return () => {
      reader?.cancel();
      port?.close();
    };
  }, [reader, port]);


  const isConnected = !!reader && !!port;
  const hasRecordedData = recordedData.length > 0;

  return (
    <main className="min-h-screen w-full bg-gray-100 flex flex-col items-center justify-center p-2 sm:p-4">
      <div className="w-full max-w-xl">
        <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-center">
          Motion Classifier using TensorFlow.js
        </h1>

        {/* Step 1: Connect & Show Data */}
        <div className="mb-8">
          <h2 className="text-lg font-bold mb-2">Step 1: Connect Device & View Data</h2>
          <button
            onClick={async () => {
              const newPort = await (navigator as any).serial.requestPort();
              await newPort.open({ baudRate: 230400 });
              const textDecoder = new TextDecoderStream();
              newPort.readable.pipeTo(textDecoder.writable);
              const newReader = textDecoder.readable.getReader();
              setPort(newPort);
              setReader(newReader);
              console.log('✅ Serial connected');
            }}
            className={`w-full sm:w-auto px-4 py-2 rounded mb-4 shadow transition ${
              isConnected
                ? 'bg-green-600 text-white'
                : 'bg-purple-600 text-white hover:bg-purple-700'
            }`}
            disabled={isConnected}
          >
            {isConnected ? 'Device Connected' : 'Connect to Accelerometer'}
          </button>
          {/* <div className="bg-gray-50 rounded-xl p-4 mb-4 w-full">
            <h3 className="font-semibold mb-2">Live Raw Data (last 10 rows)</h3>
            <table className="w-full text-sm ">
              <thead>
                <tr>
                  <th className="text-left">X (ax)</th>
                  <th className="text-left">Y (ay)</th>
                  <th className="text-left">Z (az)</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 10).map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.a}</td>
                    <td>{row.b}</td>
                    <td>{row.c}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div> */}
        </div>

        {/* Step 2: Record Data */}
        <div className="mb-8">
          <h2 className="text-lg font-bold mb-2">Step 2: Record Motion Data</h2>
          
          {/* Class Descriptions - UI only */}
          <div className="bg-blue-50 rounded-lg p-4 mb-4 border border-blue-200">
            <h3 className="font-semibold text-blue-900 mb-3">Motion Class Definitions:</h3>
            <div className="grid grid-cols-1 gap-2 text-sm">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-blue-600 text-white rounded flex items-center justify-center text-xs font-bold">0</div>
                <span><strong>Class 0:</strong> Horizontal movement (side-to-side motion)</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-green-600 text-white rounded flex items-center justify-center text-xs font-bold">1</div>
                <span><strong>Class 1:</strong> Vertical movement (up-and-down motion)</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-yellow-600 text-white rounded flex items-center justify-center text-xs font-bold">2</div>
                <span><strong>Class 2:</strong> Still/Stationary (device at rest)</span>
              </div>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-2 mb-4">
            <button
              className={`flex-1 px-3 py-2 rounded ${selectedClass === 0 ? 'bg-blue-600 text-white' : 'bg-white border'}`}
              onClick={() => { setSelectedClass(0); selectedClassRef.current = 0; }}
              disabled={!isConnected || recording}
            >
              Class 0
            </button>
            <button
              className={`flex-1 px-3 py-2 rounded ${selectedClass === 1 ? 'bg-green-600 text-white' : 'bg-white border'}`}
              onClick={() => { setSelectedClass(1); selectedClassRef.current = 1; }}
              disabled={!isConnected || recording}
            >
              Class 1
            </button>
            <button
              className={`flex-1 px-3 py-2 rounded ${selectedClass === 2 ? 'bg-yellow-600 text-white' : 'bg-white border'}`}
              onClick={() => { setSelectedClass(2); selectedClassRef.current = 2; }}
              disabled={!isConnected || recording}
            >
              Class 2
            </button>
          </div>
          <div className="flex gap-2 mb-4">
            <button
              className={`px-4 py-2 rounded ${recording ? 'bg-red-600 text-white' : 'bg-gray-200'}`}
              onClick={() => { setRecording(true); recordingRef.current = true; }}
              disabled={!isConnected || recording || selectedClassRef.current === null}
            >
              Start Recording
            </button>
            <button
              className="px-4 py-2 rounded bg-gray-200"
              onClick={() => { setRecording(false); recordingRef.current = false; }}
              disabled={!recording}
            >
              Stop Recording
            </button>
            <button
              className="px-4 py-2 rounded bg-blue-500 text-white"
              onClick={() => {
                if (recordedData.length === 0) return;
                const csvRows = [
                  'class,x,y,z',
                  ...recordedData.map(d => `${d.classLabel},${d.ax},${d.ay},${d.az}`)
                ];
                const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'motion-data.csv';
                a.click();
                URL.revokeObjectURL(url);
              }}
              disabled={recordedData.length === 0}
            >
              Download CSV
            </button>
          </div>
          
        </div>

        {/* Step 3: Train Model */}
        <div className="mb-8">
          <h2 className="text-lg font-bold mb-2">Step 3: Train Model</h2>
         
          <a
            href="/train"
            className="block w-full text-center bg-green-500 text-white px-4 py-2 rounded mb-4 shadow hover:bg-green-600 transition"
          >
            Go to Model Training Page
          </a>
        </div>

        {/* Step 4: Motion Detection */}
        <div className="mb-8">
          <h2 className="text-lg font-bold mb-2">Step 4: Motion Detection</h2>
          {model ? (
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <div className="text-center">
                <div className="mb-4">
                  <span className="text-sm text-gray-600">Current Motion:</span>
                </div>
                <div className={`inline-flex items-center gap-3 px-6 py-4 rounded-xl ${getMotionColor(motion || '')}`}>
                  {getMotionIcon(motion || '')}
                  <span className="text-xl font-bold">
                    {motion || 'No motion detected'}
                  </span>
                </div>
              </div>
              {/* <div className="mt-6 pt-4 border-t">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-xs text-gray-500">X-Axis</div>
                    <div className="font-mono text-lg">{a.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Y-Axis</div>
                    <div className="font-mono text-lg">{b.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">Z-Axis</div>
                    <div className="font-mono text-lg">{c.toFixed(2)}</div>
                  </div>
                </div>
              </div> */}
            </div>
          ) : (
            <div className="bg-gray-100 rounded-xl p-6 text-center">
              <p className="text-gray-600">No model loaded. Please train a model first.</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
