'use client'
import { useState } from 'react';
import { createAndTrainModelFromCSV } from '../../lib/trainModel';

export default function TrainModelPage() {
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [epochLogs, setEpochLogs] = useState<string[]>([]);

  return (
    <main className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-2 sm:p-4">
      <div className="w-full max-w-xl">
        <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-center">
          Train Motion Classifier Model
        </h1>
        <div className="bg-white rounded-xl shadow p-4 mb-6">
          <h2 className="text-lg font-bold mb-2">How to Train</h2>
          <ol className="list-decimal ml-6 text-gray-700 space-y-1">
            <li>Prepare your motion data as a CSV file (download from main page).</li>
            <li>Upload the CSV file below.</li>
            <li>Wait for training to complete. Progress will be shown.</li>
            <li>The trained model will be saved locally and used for predictions.</li>
          </ol>
        </div>
        <input
          type="file"
          accept=".csv"
          className="mb-4"
          disabled={training}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setTraining(true);
            setProgress('Training started...');
            setEpochLogs([]);
            const text = await file.text();
            const trainedModel = await createAndTrainModelFromCSV(text, (epoch, logs) => {
              const logMsg = `Epoch ${epoch + 1}: loss=${logs.loss?.toFixed(4)}, accuracy=${(logs.acc ? (logs.acc * 100).toFixed(2) : 'N/A')}%`;
              setEpochLogs(prev => [...prev, logMsg]);
              setProgress(logMsg);
            });
            await trainedModel.save('indexeddb://motion-model');
            setTraining(false);
            setProgress('✅ Training complete! Model saved.');
          }}
        />
        {progress && (
          <div className="mb-4 text-center text-blue-600 font-semibold">{progress}</div>
        )}
        {epochLogs.length > 0 && (
          <div className="bg-gray-50 rounded-xl p-4 mb-4 max-h-64 h-64 overflow-y-auto">
            <h3 className="font-semibold mb-2">Training Progress</h3>
            <ul className="text-xs font-mono text-gray-700 space-y-1">
              {epochLogs.map((log, i) => (
                <li key={i}>{log}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </main>
  );
}