'use client'
import { useEffect, useState, useRef } from 'react';
import { createAndTrainModelFromCSV, calculateMotionFeatures as extractFeatures } from '../lib/trainModel';
import * as tf from '@tensorflow/tfjs';
import { Activity, Zap, Target, TrendingUp, Bluetooth, BluetoothOff, RotateCw } from 'lucide-react';

// Add global type declarations for Web Bluetooth API
declare global {
  interface BluetoothRemoteGATTCharacteristic extends EventTarget {
    value?: DataView;
    startNotifications(): Promise<BluetoothRemoteGATTCharacteristic>;
    addEventListener(type: string, listener: (event: Event) => void): void;
  }
  interface BluetoothDevice extends EventTarget {
    gatt?: BluetoothRemoteGATTServer;
    name?: string;
    addEventListener(type: string, listener: (event: Event) => void): void;
  }
  interface BluetoothRemoteGATTServer {
    connected: boolean;
    connect(): Promise<BluetoothRemoteGATTServer>;
    disconnect(): void;
    getPrimaryService(service: string): Promise<BluetoothRemoteGATTService>;
  }
  interface BluetoothRemoteGATTService {
    getCharacteristic(characteristic: string): Promise<BluetoothRemoteGATTCharacteristic>;
  }
}

export default function Home() {
  // @ts-ignore: BluetoothDevice is available in browsers with Web Bluetooth API
  const [device, setDevice] = useState<BluetoothDevice | null>(null);
  // @ts-ignore: BluetoothRemoteGATTCharacteristic is available in browsers with Web Bluetooth API
  const [characteristic, setCharacteristic] = useState<BluetoothRemoteGATTCharacteristic | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [a, setA] = useState(0); // ax
  const [b, setB] = useState(0); // ay
  const [c, setC] = useState(0); // az
  const [motion, setMotion] = useState<string | null>(null);
  const [history, setHistory] = useState<{ a: number, b: number, c: number }[]>([]);
  const [recording, setRecording] = useState(false);
  const [selectedClass, setSelectedClass] = useState<number | null>(null);
  const [recordedData, setRecordedData] = useState<{ classLabel: number, features?: number[], ax?: number, ay?: number, az?: number }[]>([]);
  const [dummy, setDummy] = useState(0); // for re-render
  const recordingRef = useRef(false);
  const selectedClassRef = useRef<number | null>(null);
  const labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3'];

  // BLE Configuration
  const SERVICE_UUID = '6e400001-b5a3-f393-e0a9-e50e24dcca9e';
  const CHARACTERISTIC_UUID = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';

  // Replace the dataBuffer state with a ref
  const dataBufferRef = useRef<number[][]>([]);

  useEffect(() => {
    setModel(null); // No model loaded by default, must upload CSV
  }, []);

  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('🔄 Attempting to load model from IndexedDB...');
        const loadedModel = await tf.loadLayersModel('indexeddb://motion-model');
        setModel(loadedModel);
        console.log('✅ Model loaded from local storage');
        console.log('Model summary:', loadedModel.summary());
      } catch (err) {
        setModel(null);
        console.log('❌ No saved model found. Please upload a CSV to train.');
        console.error('Model loading error:', err);
      }
    };
    loadModel();
  }, []);

  // Handle BLE data reception
  const handleBLEData = (event: Event) => {
    const target = event.target as BluetoothRemoteGATTCharacteristic;
    const value = new TextDecoder().decode(target.value);
    
    try {
      const [ax, ay, az] = value.trim().split(',').map(Number);
      if ([ax, ay, az].some(v => isNaN(v))) {
        return;
      }

      setA(ax);
      setB(ay);
      setC(az);

      // Update buffer using ref (persists across renders)
      dataBufferRef.current = [...dataBufferRef.current, [ax, ay, az]].slice(-10);
      const finalBuffer = dataBufferRef.current;
      
      // Log buffer status
      console.log(`Buffer size: ${finalBuffer.length}/10 needed for prediction`);
      console.log('Current buffer:', finalBuffer);

      setHistory(prev => [
        { a: ax, b: ay, c: az },
        ...prev.slice(0, 19)
      ]);

      // Use the buffer length directly
      if (model && finalBuffer.length >= 10) {
        console.log('Attempting prediction with buffer:', finalBuffer);
        
        try {
          // Extract features using the current buffer
          const features = extractFeatures(finalBuffer);
          console.log('Extracted features:', features);
          console.log('Features length:', features.length);
          
          const input = tf.tensor2d([features]);
          const output = model.predict(input) as tf.Tensor;
          const result = output.dataSync();
          const predictedIndex = result.indexOf(Math.max(...result));
          const confidence = Math.max(...result);
          
          console.log('Prediction results:', result);
          console.log('Predicted class:', predictedIndex, 'with confidence:', confidence);
          
          // Lower the confidence threshold for testing
          if (confidence > 0.3) {
            setMotion(labels[predictedIndex]);
            console.log('Motion set to:', labels[predictedIndex]);
          } else {
            console.log('Confidence too low:', confidence);
          }
          
          input.dispose();
          output.dispose();
        } catch (featureError) {
          console.warn('Feature extraction error:', featureError);
        }
      } else {
        console.log('Not enough data for prediction. Buffer length:', finalBuffer.length, 'Model loaded:', !!model);
      }

      // Recording logic
      if (recordingRef.current && selectedClassRef.current !== null) {
        // Use a sliding window of the last 10 readings
        const currentWindow = [...dataBufferRef.current, [ax, ay, az]].slice(-10);
        if (currentWindow.length === 10) {
          const features = extractFeatures(currentWindow);
          setRecordedData(prev => [
            ...prev,
            {
              classLabel: selectedClassRef.current!,
              features // Store computed features
            }
          ]);
        }
      }

    } catch (err) {
      console.warn('BLE data parsing error:', err);
    }
  };

  // Connect to BLE device
  const connectBLE = async () => {
    try {
      const bleDevice = await (navigator as Navigator & { bluetooth: any }).bluetooth.requestDevice({
        filters: [{ name: 'ESP32C6_Accel' }],
        optionalServices: [SERVICE_UUID]
      });

      const server = await bleDevice.gatt?.connect();
      if (!server) throw new Error('Failed to connect to GATT server');

      const service = await server.getPrimaryService(SERVICE_UUID);
      const char = await service.getCharacteristic(CHARACTERISTIC_UUID);

      await char.startNotifications();
      char.addEventListener('characteristicvaluechanged', handleBLEData);

      setDevice(bleDevice);
      setCharacteristic(char);
      console.log('✅ BLE connected to ESP32');

      // Handle disconnection
      bleDevice.addEventListener('gattserverdisconnected', () => {
        setDevice(null);
        setCharacteristic(null);
        console.log('🔌 BLE disconnected');
      });

    } catch (error) {
      console.error('BLE connection failed:', error);
      alert('Failed to connect to BLE device. Make sure your ESP32 is powered on and nearby.');
    }
  };

  // Disconnect BLE
  const disconnectBLE = () => {
    if (device?.gatt?.connected) {
      device.gatt.disconnect();
    }
    setDevice(null);
    setCharacteristic(null);
  };

  const getMotionIcon = (motion: string) => {
    switch (motion) {
      case 'Class 0': return <TrendingUp className="w-6 h-6" />;
      case 'Class 1': return <Activity className="w-6 h-6" />;
      case 'Class 2': return <Target className="w-6 h-6" />;
      case 'Class 3': return <RotateCw className="w-6 h-6" />;
      default: return <Zap className="w-6 h-6" />;
    }
  };

  const getMotionColor = (motion: string) => {
    switch (motion) {
      case 'Class 0': return 'text-blue-600 bg-blue-50';
      case 'Class 1': return 'text-blue-700 bg-blue-100';
      case 'Class 2': return 'text-blue-800 bg-blue-200';
      case 'Class 3': return 'text-purple-600 bg-purple-50';
      default: return 'text-blue-600 bg-blue-50';
    }
  };

  useEffect(() => {
    selectedClassRef.current = selectedClass;
  }, [selectedClass]);

  useEffect(() => {
    return () => {
      disconnectBLE();
    };
  }, []);

  const isConnected = !!device && !!characteristic && device.gatt?.connected;
  const hasRecordedData = recordedData.length > 0;

  return (
    <main className="min-h-screen w-full bg-gray-100 flex flex-col items-center justify-center p-2 sm:p-4">
      <div className="w-full max-w-xl">
        <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-center">
          Motion Classifier using TensorFlow.js
        </h1>

        {/* Step 1: Connect BLE Device */}
        <div className="mb-1">
          <h2 className="text-lg font-bold mb-2 text-center">Step 1: Connect Device & View Data</h2>
          <div className="flex justify-center">
            <button
              onClick={isConnected ? disconnectBLE : connectBLE}
              className={`w-full sm:w-auto px-4 py-2 rounded mb-4 shadow transition flex items-center justify-center gap-2 ${isConnected
                  ? 'bg-green-600 text-white hover:bg-green-700'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
            >
              {isConnected ? (
                <>
                  <Bluetooth className="w-5 h-5" />
                  Device Connected (Click to Disconnect)
                </>
              ) : (
                <>
                  <BluetoothOff className="w-5 h-5" />
                  Connect to ESP32 BLE
                </>
              )}
            </button>
          </div>

          {/* {isConnected && (
            <div className="bg-green-50 rounded-lg p-4 mb-4 border border-green-200">
              <div className="flex items-center gap-2 text-green-700 mb-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="font-medium">Live BLE Data Stream</span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-gray-500 mb-1">X-Axis</div>
                  <div className="font-mono text-lg font-bold">{a.toFixed(2)}</div>
                </div>
                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-gray-500 mb-1">Y-Axis</div>
                  <div className="font-mono text-lg font-bold">{b.toFixed(2)}</div>
                </div>
                <div className="bg-white rounded-lg p-3">
                  <div className="text-xs text-gray-500 mb-1">Z-Axis</div>
                  <div className="font-mono text-lg font-bold">{c.toFixed(2)}</div>
                </div>
              </div>
            </div>
          )} */}
        </div>

        {/* Step 2: Record Data */}
        <div className="mb-8">
          <h2 className="text-lg font-bold mb-2 text-center">Step 2: Record Motion Data</h2>

          {/* Class Descriptions - UI only */}
          <div className="bg-blue-50 rounded-lg p-4 mb-4 border border-blue-200">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex flex-row items-center gap-3">
                <div className="w-6 h-6 bg-blue-600 text-white rounded flex items-center justify-center text-xs font-bold">0</div>
                <span><strong>Class 0:</strong> Horizontal movement</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-green-600 text-white rounded flex items-center justify-center text-xs font-bold">1</div>
                <span><strong>Class 1:</strong> Vertical movement</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-yellow-600 text-white rounded flex items-center justify-center text-xs font-bold">2</div>
                <span><strong>Class 2:</strong> Still/Stationary</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-purple-600 text-white rounded flex items-center justify-center text-xs font-bold">3</div>
                <span><strong>Class 3:</strong> Circular motion</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
            <button
              className={`px-3 py-2 rounded ${selectedClass === 0 ? 'bg-blue-600 text-white' : 'bg-white border'}`}
              onClick={() => { setSelectedClass(0); selectedClassRef.current = 0; }}
              disabled={!isConnected || recording}
            >
              Class 0
            </button>
            <button
              className={`px-3 py-2 rounded ${selectedClass === 1 ? 'bg-green-600 text-white' : 'bg-white border'}`}
              onClick={() => { setSelectedClass(1); selectedClassRef.current = 1; }}
              disabled={!isConnected || recording}
            >
              Class 1
            </button>
            <button
              className={`px-3 py-2 rounded ${selectedClass === 2 ? 'bg-yellow-600 text-white' : 'bg-white border'}`}
              onClick={() => { setSelectedClass(2); selectedClassRef.current = 2; }}
              disabled={!isConnected || recording}
            >
              Class 2
            </button>
            <button
              className={`px-3 py-2 rounded ${selectedClass === 3 ? 'bg-purple-600 text-white' : 'bg-white border'}`}
              onClick={() => { setSelectedClass(3); selectedClassRef.current = 3; }}
              disabled={!isConnected || recording}
            >
              Class 3
            </button>
          </div>
          <div className="flex gap-2 mb-4 justify-center">
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
                const featureCount = recordedData[0].features?.length || 0;
                // Update your CSV download logic
                const csvContent = [
                  ['class', ...Array.from({length: featureCount}, (_, i) => `feature_${i}`)].join(','),
                  ...recordedData.map(d => [d.classLabel, ...(d.features ?? [])].join(','))
                ].join('\n');
                const blob = new Blob([csvContent], { type: 'text/csv' });
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

          {hasRecordedData && (
            <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
              <div className="text-gray-700 font-medium">
                📊 Total samples: {recordedData.length}
              </div>
              <div className="text-sm text-gray-600 mt-1">
                Class 0: {recordedData.filter(d => d.classLabel === 0).length} •
                Class 1: {recordedData.filter(d => d.classLabel === 1).length} •
                Class 2: {recordedData.filter(d => d.classLabel === 2).length} •
                Class 3: {recordedData.filter(d => d.classLabel === 3).length}
              </div>
            </div>
          )}
        </div>

        {/* Step 3: Train Model */}
        <div className="mb-8">
          <h2 className="text-lg font-bold mb-2 text-center">Step 3: Train Model</h2>

          <a
            href="/train"
            className="block w-full text-center bg-green-500 text-white px-4 py-2 rounded mb-4 shadow hover:bg-green-600 transition"
          >
            Go to Model Training Page
          </a>
        </div>

        {/* Step 4: Motion Detection */}
        <div className="mb-2">
          <h2 className="text-lg font-bold mb-2 text-center">Step 4: Motion Detection</h2>
          {model ? (
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <div className="text-center">
                <div className={`inline-flex items-center gap-3 px-6 py-4 rounded-xl ${getMotionColor(motion || '')}`}>
                  {getMotionIcon(motion || '')}
                  <span className="text-xl font-bold">
                    {motion || 'No motion detected'}
                  </span>
                </div>
              </div>
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

function processCSV(csvText: string): { xs: tf.Tensor2D; ys: tf.Tensor2D } {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',');
  const dataLines = lines.slice(1);

  const allFeatures: number[][] = [];
  const allLabels: number[][] = [];

  for (const line of dataLines) {
    const values = line.split(',').map(Number);
    const classStr = values[0];
    const features = values.slice(1); // All columns except first are features
    
    allFeatures.push(features);
    
    // One-hot encoding for 4 classes
    if (classStr === 0) allLabels.push([1, 0, 0, 0]);      // horizontal
    else if (classStr === 1) allLabels.push([0, 1, 0, 0]); // vertical  
    else if (classStr === 2) allLabels.push([0, 0, 1, 0]); // still
    else allLabels.push([0, 0, 0, 1]);                     // circular
  }

  return {
    xs: tf.tensor2d(allFeatures),
    ys: tf.tensor2d(allLabels)
  };
}

