# 🧠 Real-Time Motion Classifier using TensorFlow.js & Advanced Feature Engineering

This project is a **real-time motion classification system** that uses **advanced accelerometer sensor analysis** to predict motion states: **horizontal**, **vertical**, **still**, and **circular**. It leverages sophisticated **feature extraction algorithms** and a **TensorFlow.js neural network** running entirely in the browser, with live sensor input through the **Web Serial API**.

---

## 🎯 Project Highlights

- 🔌 Connects to external motion sensors (e.g., MPU6050) via Web Serial
- 📡 Reads live accelerometer data in real time
- 🧠 **Orientation-independent motion classification** using advanced feature engineering
- 📊 **23+ extracted features** including velocity, acceleration, variance, correlations, and magnitude analysis
- 🤖 Deep neural network with **64→32→16→4 architecture** and dropout regularization
- 🖥️ Displays live sensor values and predictions in a smooth web UI
- ⚙️ Built using **Next.js + TypeScript + TensorFlow.js**
- 🚀 No backend required — everything runs inside the browser
- 🔄 **Works regardless of device orientation** through motion pattern analysis

---

## 🧪 Enhanced Motion Detection

### Motion Classes
- **Horizontal** - Side-to-side movement
- **Vertical** - Up-and-down movement  
- **Still** - No significant movement (orientation-independent)
- **Circular** - Rotational or circular motion patterns

### Advanced Features
- **Temporal Analysis**: Uses sliding windows of 10 consecutive readings
- **Statistical Features**: Mean, variance, range analysis per axis
- **Motion Dynamics**: Velocity and acceleration (jerk) detection
- **Cross-Axis Analysis**: Correlation patterns for circular motion detection
- **Magnitude Processing**: Overall movement intensity regardless of orientation
- **Dominant Axis Detection**: Identifies primary movement direction

---

## 🧪 Use Cases

- Gesture-based interfaces and controls  
- Smart wearable motion detection  
- Physical activity monitoring and fitness tracking
- Educational demos for real-time ML in the browser  
- IoT or robotics motion understanding
- Orientation-independent device control systems

---

## 🏗️ Technical Architecture

### Machine Learning Model
- **Type**: Sequential Neural Network (TensorFlow.js)
- **Architecture**: Dense layers with dropout regularization
- **Input**: 23 engineered features (not raw coordinates)
- **Output**: 4-class softmax classification
- **Training**: Adam optimizer with L2 regularization

### Feature Engineering Pipeline
```typescript
Raw Accelerometer Data → Sliding Window (10 samples) → Feature Extraction → Neural Network → Motion Class
```

### Key Features Extracted
1. **Current Values**: Latest x, y, z readings
2. **Statistical Analysis**: Mean, variance, range per axis
3. **Motion Dynamics**: Velocity and acceleration calculations
4. **Axis Relationships**: Cross-correlation analysis
5. **Movement Intensity**: Magnitude and magnitude variation
6. **Dominant Direction**: Activity ratios between axes

---

## 📂 Folder Structure

```bash
├── src/
│   ├── lib/
│   │   └── trainModel.ts        # Advanced feature extraction & TensorFlow.js model
│   └── app/
│       └── page.tsx             # UI + live serial stream + prediction logic
├── README.md                    # You're here!
└── package.json                 # Dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js and npm
- Chrome/Edge browser (for Web Serial API)
- MPU6050 or compatible accelerometer sensor
- Arduino or microcontroller for sensor data transmission

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/gyroscope-motion-classifier

# Install dependencies
npm install

# Run development server
npm run dev
```

### Usage
1. Connect your accelerometer sensor via USB/Serial
2. Open the web app and click "Connect to Serial"
3. Select motion class and record training data in various orientations
4. Train the model using the recorded data
5. Test real-time classification with different device orientations

---

## 🔬 Technical Details

### Why Feature Engineering?
This project moved beyond raw coordinate classification to **feature-based motion analysis** to solve the **orientation dependence problem**. Traditional models fail when the device is rotated because they rely on absolute x, y, z values. Our approach extracts **motion characteristics** that remain consistent regardless of device orientation.

### Model Performance
- **Orientation Independence**: Works in any device rotation
- **Real-time Processing**: <10ms prediction latency
- **High Accuracy**: Enhanced feature set improves classification reliability
- **Robust Training**: Dropout and L2 regularization prevent overfitting

---

## Acknowledgments

Special thanks to **[Deepak Khatri](https://github.com/lorforlinux)** and **[Krishnanshu Mittal](https://github.com/CIumsy)** for their invaluable guidance and support throughout this project — from building the  ML pipeline to handling real-time sensor data processing and feature engineering.  
Their expertise helped shape the AI architecture and hardware integration aspects of this work.

## 👩‍💻 Author

- **Ritika Mishra** - Developer 

---

## 🤝 Contributing

Contributions, suggestions, and collaborations are welcome!

If you'd like to improve the motion model, add support for more sensors, enhance feature extraction, or build new ML capabilities:

1. Fork the repository  
2. Create your feature branch: `git checkout -b feature-name`  
3. Commit your changes: `git commit -m 'Add feature'`  
4. Push to the branch: `git push origin feature-name`  
5. Open a Pull Request

### Areas for Contribution
- Additional motion classes (walking, running, etc.)
- More sophisticated feature engineering
- Real-time model retraining capabilities
- Support for additional sensor types
- Performance optimizations

For major changes, please open an issue first to discuss what you'd like to change.

---

⭐ Don't forget to **star the repo** if you find it helpful!

## 🏷️ Keywords

`tensorflow.js` `machine-learning` `motion-detection` `accelerometer` `feature-engineering` `real-time-classification` `web-serial-api` `orientation-independent` `neural-networks` `browser-ml`
