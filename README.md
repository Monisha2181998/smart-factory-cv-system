# Smart Factory Real-Time Object Detection Pipeline

**Author:** Monisha Ravi Kumar  
**Contact:** monisharavikumar21@gmail.com | [GitHub](https://github.com/Monisha2181998)  
**Application:** Wissenschaftliche/r Mitarbeiter/in — Computer Vision / Data Science  
**Professur Smart Systems Integration, TU Chemnitz**

---

## Overview

This project implements a **real-time object detection and performance benchmarking pipeline** targeting industrial smart environment monitoring. It demonstrates end-to-end Computer Vision system design: from data acquisition and preprocessing through algorithm implementation, real-world inference, and quantitative performance validation — directly mirroring the research tasks outlined in the SSI Professur position.

The pipeline provides:
- **Python implementation** using YOLOv8 (Ultralytics) + OpenCV for video I/O and visualisation
- **C++ implementation** using OpenCV DNN module with ONNX model export for embedded/edge deployment
- **Automated benchmark suite** comparing model variants across latency (avg, P95, P99), throughput (FPS), and detection stability

---

## Motivation & Research Context

Industrial smart environments require robust, low-latency vision systems capable of detecting personnel, equipment, and anomalies in real time. Key engineering challenges include:

- Selecting the right model size for the latency/accuracy trade-off on edge hardware
- Validating inference stability under continuous stream conditions (P95/P99 latency)
- Designing a software architecture that separates data ingestion, model inference, and logging cleanly
- Enabling both Python-based rapid prototyping and C++ deployment on resource-constrained systems

This project addresses all four challenges with a clean, extensible architecture.

---

## Project Structure

```
smart_factory_od/
├── src/
│   ├── detector.py        # Main pipeline: webcam / video / image dir
│   └── benchmark.py       # Model comparison benchmark suite
├── cpp/
│   └── detector.cpp       # C++ / OpenCV DNN inference engine
├── results/               # Auto-generated outputs (JSON, CSV, PNG, video)
├── data/
│   └── sample_images/     # Drop test images here
├── requirements.txt
└── README.md
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Input Sources                         │
│   Webcam (live)  │  Video file  │  Image directory      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Pre-processing (OpenCV)                    │
│   Resize → Normalise → Blob creation (640×640)          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Inference Engine (YOLOv8 / ONNX)              │
│   Python: Ultralytics YOLO  │  C++: OpenCV DNN module   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         Post-processing & NMS                           │
│   Confidence filtering → IoU-based NMS → Detection list │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
   ┌─────────────────┐   ┌──────────────────────┐
   │  Visualisation  │   │  Metrics Logger       │
   │  Bounding boxes │   │  Latency, FPS, counts │
   │  HUD overlay    │   │  → JSON + CSV export  │
   └─────────────────┘   └──────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/Monisha2181998/smart-factory-detection.git
cd smart_factory_od

# Python environment
pip install -r requirements.txt

# C++ build (requires OpenCV 4.x with DNN module)
cd cpp
g++ -std=c++17 -O2 detector.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -o smart_factory_detector
```

---

## Usage

### Python — Live Detection

```bash
# Webcam
python src/detector.py --source 0 --model n --conf 0.35

# Video file
python src/detector.py --source factory_floor.mp4 --model s

# Image directory
python src/detector.py --source data/sample_images/

# Headless (no display, e.g. server)
python src/detector.py --source factory.mp4 --no-show --model m
```

### Python — Model Benchmark

```bash
# Compare nano, small, medium on 50 frames
python src/benchmark.py --models n s m --frames 50 --source 0

# On video file
python src/benchmark.py --models n s --frames 100 --source factory.mp4

# Fully offline synthetic benchmark
python src/benchmark.py --source synthetic --frames 50
```

### C++ — ONNX Inference

```bash
# Export model first (Python)
yolo export model=yolov8n.pt format=onnx

# Run C++ detector
./cpp/smart_factory_detector --model yolov8n.onnx --source factory.mp4
./cpp/smart_factory_detector --model yolov8n.onnx --source 0 --no-save
```

---

## Benchmark Results

> Evaluated on 50-frame test set. YOLOv8n meets real-time threshold (≥30 FPS) with lowest latency variance.

| Model    | Avg FPS | Avg Latency | P95 Latency | P99 Latency |
|----------|--------:|------------:|------------:|------------:|
| YOLOv8n  |   83.4  |    12.0 ms  |    14.8 ms  |    15.0 ms  |
| YOLOv8s  |   45.8  |    21.8 ms  |    26.6 ms  |    27.8 ms  |
| YOLOv8m  |   24.5  |    40.8 ms  |    49.0 ms  |    52.8 ms  |

**Key finding:** YOLOv8n is the optimal choice for real-time industrial monitoring on CPU hardware — it comfortably exceeds the 30 FPS threshold with low latency variance (σ = 1.8 ms), making it suitable for latency-critical safety applications.

![Benchmark Comparison](results/benchmark_comparison.png)

---

## Output Files

| File | Description |
|------|-------------|
| `results/output_<session>.mp4` | Annotated output video |
| `results/benchmark_<session>.json` | Per-session inference metrics |
| `results/frame_metrics_<session>.csv` | Per-frame latency + FPS log |
| `results/benchmark_comparison.png` | Multi-model comparison chart |

---

## Key Design Decisions

**Why YOLOv8?** YOLOv8 offers ONNX export out of the box, enabling the same trained weights to run in Python (rapid prototyping) and C++ via OpenCV DNN (deployment). This single-model, dual-runtime architecture is crucial for production industrial systems.

**Why P95/P99 latency?** Average latency is misleading for real-time systems. Tail latency determines whether a safety-critical detection pipeline will meet its deadline under all conditions. The benchmark explicitly tracks P95 and P99 to surface worst-case behaviour.

**Why OpenCV DNN for C++?** OpenCV DNN requires no additional ML framework at runtime — it depends only on OpenCV, which is already present in virtually all embedded Linux and industrial vision systems.

---

## Technologies

| Layer | Python | C++ |
|-------|--------|-----|
| Detection | YOLOv8 (Ultralytics) | OpenCV DNN + ONNX |
| Video I/O | OpenCV | OpenCV |
| Visualisation | OpenCV | OpenCV |
| Logging | JSON + CSV | stdout + file |
| Build | pip / requirements.txt | g++ / pkg-config |

---

## License

MIT License — free to use, modify, and distribute.
