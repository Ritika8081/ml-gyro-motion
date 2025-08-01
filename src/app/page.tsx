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
  const labels = ['horizontal', 'vertical', 'still'];



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
      case 'horizontal': return <TrendingUp className="w-6 h-6" />;
      case 'vertical': return <Activity className="w-6 h-6" />;
      case 'still': return <Target className="w-6 h-6" />;
      default: return <Zap className="w-6 h-6" />;
    }
  };

  const getMotionColor = (motion: string) => {
    switch (motion) {
      case 'horizontal': return 'text-blue-600 bg-blue-50';
      case 'vertical': return 'text-blue-700 bg-blue-100';
      case 'still': return 'text-blue-800 bg-blue-200';
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


  return (
    <main className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-2 sm:p-4">
      <div className="w-full max-w-xl">
        <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-center">
          Motion Classifier using TensorFlow.js
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


        {/* Manual Input & Predict
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
          
        </div> */}

        {motion ? (
          <div className={`rounded-xl p-6 mb-6 ${getMotionColor(motion)}`}>
            <div className="flex items-center gap-4">
              {getMotionIcon(motion)}
              <div>
                <div className="text-sm font-medium opacity-80">Detected Motion</div>
                <div className="text-2xl font-bold capitalize">{motion}</div>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-gray-50 rounded-xl p-6 mb-6 text-center">
            <Target className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <div className="text-gray-500">No prediction yet</div>
          </div>
        )}
        <div className="flex flex-col sm:flex-row gap-2 mb-4">
          <button
            className={`flex-1 px-3 py-2 rounded ${selectedClass === 0 ? 'bg-blue-600 text-white' : 'bg-white border'}`}
            onClick={() => { setSelectedClass(0); selectedClassRef.current = 0; }}
          >
            Horizontal
          </button>
          <button
            className={`flex-1 px-3 py-2 rounded ${selectedClass === 1 ? 'bg-green-600 text-white' : 'bg-white border'}`}
            onClick={() => { setSelectedClass(1); selectedClassRef.current = 1; }}
          >
            Vertical
          </button>
          <button
            className={`flex-1 px-3 py-2 rounded ${selectedClass === 2 ? 'bg-yellow-600 text-white' : 'bg-white border'}`}
            onClick={() => { setSelectedClass(2); selectedClassRef.current = 2; }}
          >
            Still
          </button>
        </div>
        <div className="flex gap-2 mb-4">
          <button
            className={`px-4 py-2 rounded ${recording ? 'bg-red-600 text-white' : 'bg-gray-200'}`}
            onClick={() => { setRecording(true); recordingRef.current = true; }}
            disabled={recording || selectedClassRef.current === null}
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
          <button
            className="px-4 py-2 rounded bg-red-500 text-white"
            onClick={async () => {
              try {
                await tf.io.removeModel('indexeddb://motion-model');
                setModel(null);
                console.log('🗑️ Model removed from local storage.');
              } catch (err) {
                console.log('No model found in local storage to remove.');
              }
            }}
          >
            Reset Model
          </button>
        </div>
        <input
          type="file"
          accept=".csv"
          className="mb-4"
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;

            const text = await file.text();
            const trainedModel = await createAndTrainModelFromCSV(text);
            await trainedModel.save('indexeddb://motion-model'); // Save to local storage
            setModel(trainedModel);
            console.log("✅ Model trained from CSV and saved to local storage.");
          }}
        />

      </div>
    </main>
  );
}
