# 🚀 Smart Factory Real-Time Object Detection System

**Author:** Monisha Ravi Kumar  
**Contact:** monisharavikumar21@gmail.com  
**GitHub:** Monisha2181998  

---

## 📌 Overview

This project implements a real-time object detection system for industrial environments using YOLOv8. It provides a complete pipeline for video-based inference, benchmarking, and model comparison across multiple YOLO variants.

The system demonstrates an end-to-end computer vision workflow including detection, visualization, and performance evaluation.

---

## 🧠 Features

- Real-time object detection using YOLOv8 (Ultralytics)
- Supports video, webcam, and image inputs
- Benchmarking of YOLOv8n, YOLOv8s, YOLOv8m models
- FPS and latency analysis (Avg, P95, P99)
- Annotated output video generation
- JSON and CSV logging of results

---

## 📂 Project Structure

```
smart_factory_od/
├── src/
│   ├── detector.py
│   ├── benchmark.py
│
├── cpp/
│   ├── detector.cpp
│
├── data/
│   ├── videos/
│
├── results/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Monisha2181998/smart-factory-cv-system.git
cd smart_factory_od
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🎥 Run Real-Time Detection

```bash
python src/detector.py --source data/videos/factory_test.mp4 --model n --conf 0.35
```

Webcam mode:
```bash
python src/detector.py --source 0 --model n
```

---

### 📊 Run Benchmark

```bash
python src/benchmark.py --models n s m --frames 50
```

---

## 📈 Results (CPU Benchmark)

| Model     | Avg FPS | Avg Latency |
|-----------|--------|-------------|
| YOLOv8n   | 15–18   | ~50 ms      |
| YOLOv8s   | 8–10    | ~110 ms     |
| YOLOv8m   | 3–5     | ~280 ms     |

---

## 🎯 Key Insight

YOLOv8n provides the best balance between speed and accuracy for real-time industrial monitoring on CPU-based systems. It is suitable for edge deployment where low latency is critical.

---

## 🛠️ Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy
- Pandas
- Matplotlib

---

## 📦 Output Files

- Annotated output video
- Benchmark JSON report
- Frame-wise CSV logs
- Performance comparison plots

---

## 📌 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

Monisha Ravi Kumar  
📧 monisharavikumar21@gmail.com  
GitHub: Monisha2181998
