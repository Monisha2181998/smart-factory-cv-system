"""
Model Benchmarking Script — Smart Factory Detection Pipeline
Author: Monisha Ravi Kumar

Compares YOLOv8 model variants (nano, small, medium) across:
  - Inference latency (avg, P95, P99)
  - Throughput (FPS)
  - Detection count stability

Outputs: benchmark_comparison.json + benchmark_plot.png
"""

import cv2
import time
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def benchmark_model(model_size: str, frames: list, conf: float = 0.35,
                    iou: float = 0.45) -> dict:
    """Run inference on a list of frames and return timing statistics."""
    print(f"\n[BENCH] YOLOv8{model_size} — {len(frames)} frames")

    if not YOLO_AVAILABLE:
        # Demo mode: synthetic latencies
        np.random.seed(ord(model_size))
        base = {"n": 12, "s": 22, "m": 40}[model_size]
        latencies = np.random.normal(base, base * 0.15, len(frames)).tolist()
        det_counts = np.random.randint(1, 6, len(frames)).tolist()
    else:
        model = YOLO(f"yolov8{model_size}.pt")
        # Warm-up
        for f in frames[:3]:
            model(f, conf=conf, iou=iou, verbose=False)

        latencies = []
        det_counts = []
        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            res = model(frame, conf=conf, iou=iou, verbose=False)[0]
            latencies.append((time.perf_counter() - t0) * 1000)
            n = len(res.boxes) if res.boxes is not None else 0
            det_counts.append(n)
            if (i + 1) % 10 == 0:
                print(f"  frame {i+1:3d}/{len(frames)}  "
                      f"lat={latencies[-1]:.1f}ms  det={n}")

    latencies = np.array(latencies)
    return {
        "model":          f"YOLOv8{model_size}",
        "n_frames":       len(frames),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "std_latency_ms": round(float(np.std(latencies)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "avg_fps":        round(1000.0 / float(np.mean(latencies)), 2),
        "avg_detections": round(float(np.mean(det_counts)), 2),
        "latency_series": [round(x, 2) for x in latencies.tolist()],
    }


def plot_results(results: list, output_dir: Path):
    """Generate a 2x2 benchmark comparison figure."""
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib not available — skipping plot")
        return

    models   = [r["model"] for r in results]
    avg_lat  = [r["avg_latency_ms"] for r in results]
    p95_lat  = [r["p95_latency_ms"] for r in results]
    p99_lat  = [r["p99_latency_ms"] for r in results]
    avg_fps  = [r["avg_fps"] for r in results]
    avg_det  = [r["avg_detections"] for r in results]

    colours = ["#2E6DB4", "#E07B39", "#3BAA6E"]
    x = np.arange(len(models))
    w = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "YOLOv8 Model Comparison — Smart Factory Detection Pipeline\n"
        "Monisha Ravi Kumar · TU Chemnitz Application Project",
        fontsize=13, fontweight="bold", y=1.01
    )

    # 1. Average latency
    ax = axes[0, 0]
    bars = ax.bar(x, avg_lat, width=w, color=colours[:len(models)], edgecolor="white")
    ax.set_title("Average Inference Latency (ms)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Latency (ms)")
    for bar, val in zip(bars, avg_lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # 2. Latency percentiles (grouped bar)
    ax = axes[0, 1]
    w2 = 0.25
    ax.bar(x - w2, avg_lat, width=w2, color=colours[0], label="Avg", edgecolor="white")
    ax.bar(x,       p95_lat, width=w2, color=colours[1], label="P95", edgecolor="white")
    ax.bar(x + w2,  p99_lat, width=w2, color=colours[2], label="P99", edgecolor="white")
    ax.set_title("Latency Percentiles (ms)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 3. Throughput (FPS)
    ax = axes[1, 0]
    bars = ax.bar(x, avg_fps, width=w, color=colours[:len(models)], edgecolor="white")
    ax.set_title("Throughput — Average FPS", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Frames per Second")
    for bar, val in zip(bars, avg_fps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(30, color="red", linestyle="--", linewidth=1, alpha=0.6,
               label="30 FPS real-time threshold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 4. Latency over time (line)
    ax = axes[1, 1]
    for i, r in enumerate(results):
        series = r["latency_series"]
        ax.plot(series, color=colours[i], alpha=0.8,
                linewidth=1.2, label=r["model"])
    ax.set_title("Latency Over Time (per frame)", fontweight="bold")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Latency (ms)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "benchmark_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot saved → {out_path}")


def load_test_frames(source, n_frames: int = 50) -> list:
    """Load frames from webcam, video, or generate synthetic frames."""
    frames = []

    if source == "synthetic" or source is None:
        print(f"[INFO] Generating {n_frames} synthetic test frames (640x480)")
        np.random.seed(42)
        for i in range(n_frames):
            # Varied synthetic frames to simulate scene changes
            frame = np.random.randint(40, 200,
                                      (480, 640, 3), dtype=np.uint8)
            # Add rectangles to simulate objects
            for _ in range(np.random.randint(2, 6)):
                x1 = np.random.randint(0, 540)
                y1 = np.random.randint(0, 380)
                cv2.rectangle(frame,
                              (x1, y1),
                              (x1 + np.random.randint(50, 100),
                               y1 + np.random.randint(50, 100)),
                              tuple(int(c) for c in
                                    np.random.randint(100, 255, 3)),
                              -1)
            frames.append(frame)
        return frames

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {source} — using synthetic frames")
        return load_test_frames("synthetic", n_frames)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
    step  = max(1, total // n_frames)
    idx   = 0
    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    print(f"[INFO] Loaded {len(frames)} frames from source")
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Model Benchmark — Smart Factory Pipeline")
    parser.add_argument("--source", default="synthetic",
                        help="0=webcam, video path, or 'synthetic'")
    parser.add_argument("--models", nargs="+", default=["n", "s", "m"],
                        help="Model sizes to compare")
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of frames to benchmark per model")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_test_frames(args.source, args.frames)

    all_results = []
    for size in args.models:
        res = benchmark_model(size, frames, args.conf)
        all_results.append(res)

    # Print comparison table
    print("\n" + "="*65)
    print("  MODEL COMPARISON SUMMARY")
    print("="*65)
    print(f"  {'Model':<12} {'Avg FPS':>9} {'Avg Lat':>10} "
          f"{'P95 Lat':>10} {'P99 Lat':>10}")
    print("-"*65)
    for r in all_results:
        print(f"  {r['model']:<12} {r['avg_fps']:>9.1f} "
              f"{r['avg_latency_ms']:>9.1f}ms "
              f"{r['p95_latency_ms']:>9.1f}ms "
              f"{r['p99_latency_ms']:>9.1f}ms")
    print("="*65)

    # Save JSON
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_comparison_{session}.json"
    with open(json_path, "w") as f:
        json.dump({"session": session, "results": all_results}, f, indent=2)
    print(f"\n[INFO] Results saved → {json_path}")

    plot_results(all_results, out_dir)


if __name__ == "__main__":
    main()
