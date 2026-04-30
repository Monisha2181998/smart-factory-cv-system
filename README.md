🚀 Smart Factory Real-Time Object Detection System

Author: Monisha Ravi Kumar
Contact: monisharavikumar21@gmail.com

GitHub: Monisha2181998

📌 Overview

This project implements a real-time object detection system for industrial environments using YOLOv8. It includes a complete pipeline for video-based inference, performance benchmarking, and model comparison across multiple YOLO variants.

The system supports:

Real-time object detection on factory videos/webcam
Benchmarking YOLOv8n, YOLOv8s, YOLOv8m
FPS and latency analysis (avg, P95, P99)
Annotated output video generation
🧠 Features
Real-time detection using YOLOv8 (Ultralytics)
Video, webcam, and image input support
Performance benchmarking suite
JSON + CSV logging of results
Annotated output video generation
📂 Project Structure
smart_factory_od/
├── src/
│   ├── detector.py
│   ├── benchmark.py
├── cpp/
│   ├── detector.cpp
├── data/
│   ├── videos/
├── results/
├── requirements.txt
├── README.md
⚙️ Installation
git clone https://github.com/Monisha2181998/smart-factory-cv-system.git
cd smart_factory_od
pip install -r requirements.txt
▶️ Usage
🎥 Real-time Detection
python src/detector.py --source data/videos/factory_test.mp4 --model n --conf 0.35
📊 Benchmark Models
python src/benchmark.py --models n s m --frames 50
📈 Results (CPU Benchmark)
Model	Avg FPS	Avg Latency
YOLOv8n	15–18	~50 ms
YOLOv8s	8–10	~110 ms
YOLOv8m	3–5	~280 ms
