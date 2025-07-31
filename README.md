# 🧠 Real-Time Motion Classifier using TensorFlow.js & Gyroscope

This project is a **real-time motion classification system** that uses **gyroscope or accelerometer sensor data** to predict motion states: **horizontal**, **vertical**, or **still**. It leverages a **TensorFlow.js machine learning model** running entirely in the browser and integrates live sensor input through the **Web Serial API**.

---

## 🎯 Project Highlights

- 🔌 Connects to external motion sensors (e.g., MPU6050) via Web Serial
- 📡 Reads live gyroscope/accelerometer data in real time
- 🧠 Classifies motion using a trained TensorFlow.js model (client-side)
- 🖥️ Displays live sensor values and predictions in a smooth web UI
- ⚙️ Built using **Next.js + Javascript + TF.js**
- 🚀 No backend required — everything runs inside the browser

---

## 🧪 Use Cases

- Gesture-based interfaces and controls  
- Smart wearable motion detection  
- Physical activity monitoring  
- Educational demos for real-time ML in the browser  
- IoT or robotics motion understanding

---

## 📂 Folder Structure

```bash
├── lib/
│   └── trainModel.ts        # TensorFlow.js model creation & training
├── app/
│   └── page.tsx             # UI + live serial stream + prediction logic
└── README.md                # You're here!

```

## 👩‍💻 Author

- Ritika Mishra

---

## 🤝 Contributing

Contributions, suggestions, and collaborations are welcome!

If you’d like to improve the motion model, add support for more sensors, or build new ML features:

1. Fork the repository  
2. Create your feature branch: `git checkout -b feature-name`  
3. Commit your changes: `git commit -m 'Add feature'`  
4. Push to the branch: `git push origin feature-name`  
5. Open a Pull Request

For major changes, please open an issue first to discuss what you’d like to change.

---

⭐ Don’t forget to **star the repo** if you find it helpful!
