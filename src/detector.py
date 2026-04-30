"""
Real-Time Object Detection Pipeline for Industrial Smart Environment Monitoring
Author: Monisha Ravi Kumar
Description:
    Implements a configurable YOLOv8-based detection pipeline supporting
    webcam, video file, and image directory inputs. Logs per-frame inference
    metrics (FPS, latency, class counts) and exports a structured JSON/CSV
    benchmark report for quantitative analysis.
"""

import cv2
import time
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ── Industrial-relevant COCO classes (subset) ──────────────────────────────
INDUSTRIAL_CLASSES = {
    0: "person",       # worker / personnel
    2: "car",          # forklift / vehicle (proxy)
    3: "motorcycle",   # mobility device
    7: "truck",        # heavy vehicle
    24: "backpack",    # personal equipment
    26: "handbag",     # carried object
    39: "bottle",      # container
    41: "cup",         # container
    56: "chair",       # workstation furniture
    57: "couch",       # rest area
    58: "potted plant",
    60: "dining table",# work surface (proxy)
    63: "laptop",      # control station
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    73: "book",        # documentation
    76: "scissors",    # tool (proxy)
    79: "toothbrush",  # small tool (proxy)
}


class IndustrialDetector:
    """
    YOLOv8 detection pipeline with per-frame benchmarking and
    structured result logging for industrial environment monitoring.
    """

    def __init__(self, model_size: str = "n", conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45, output_dir: str = "results"):
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size

        # Metrics storage
        self.frame_metrics = []
        self.class_counts = defaultdict(int)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"[INFO] Initialising IndustrialDetector (YOLOv8{model_size})")
        print(f"[INFO] Confidence threshold : {conf_threshold}")
        print(f"[INFO] IoU threshold        : {iou_threshold}")
        print(f"[INFO] Results directory    : {self.output_dir}")

        if YOLO_AVAILABLE:
            self.model = YOLO(f"yolov8{model_size}.pt")
            print(f"[INFO] Model loaded: yolov8{model_size}.pt")
        else:
            self.model = None
            print("[WARN] ultralytics not available — running in DEMO mode")

    # ── Core inference ─────────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray):
        """Run YOLOv8 inference and return (results, latency_ms)."""
        t0 = time.perf_counter()
        if self.model:
            results = self.model(frame, conf=self.conf, iou=self.iou,
                                 verbose=False)[0]
        else:
            results = None
        latency = (time.perf_counter() - t0) * 1000
        return results, latency

    def _draw(self, frame: np.ndarray, results, fps: float) -> np.ndarray:
        """Draw bounding boxes, labels, and HUD onto frame."""
        overlay = frame.copy()

        if results and results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = results.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Colour by class (deterministic)
                colour = self._class_colour(cls_id)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
                tag = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(overlay, tag, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # HUD
        hud = f"FPS: {fps:5.1f}  |  YOLOv8{self.model_size}  |  conf={self.conf}"
        cv2.rectangle(overlay, (0, 0), (len(hud) * 9 + 10, 28), (20, 20, 20), -1)
        cv2.putText(overlay, hud, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)

        n_det = len(results.boxes) if (results and results.boxes is not None) else 0
        cv2.putText(overlay, f"Detections: {n_det}", (6, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        return overlay

    @staticmethod
    def _class_colour(cls_id: int):
        np.random.seed(cls_id * 7 + 13)
        return tuple(int(c) for c in np.random.randint(80, 230, 3))

    def _log_frame(self, frame_idx: int, latency: float, fps: float, results):
        """Accumulate per-frame metrics."""
        det_classes = {}
        if results and results.boxes is not None:
            for box in results.boxes:
                name = results.names.get(int(box.cls[0]), "unknown")
                det_classes[name] = det_classes.get(name, 0) + 1
                self.class_counts[name] += 1

        self.frame_metrics.append({
            "frame":      frame_idx,
            "latency_ms": round(latency, 2),
            "fps":        round(fps, 2),
            "n_detections": sum(det_classes.values()),
            "classes":    det_classes,
        })

    # ── Input modes ────────────────────────────────────────────────────────

    def run_video(self, source, show: bool = True, save: bool = True,
                  max_frames: int = 0):
        """Process video file or webcam (source=0)."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        writer = None
        if save:
            out_path = self.output_dir / f"output_{self.session_id}.mp4"
            writer = cv2.VideoWriter(str(out_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     src_fps, (w, h))
            print(f"[INFO] Saving output to: {out_path}")

        frame_idx = 0
        fps_smooth = 0.0
        t_prev = time.perf_counter()

        print("[INFO] Running — press Q to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            results, latency = self._infer(frame)

            t_now   = time.perf_counter()
            fps_raw = 1.0 / max(t_now - t_prev, 1e-6)
            fps_smooth = 0.8 * fps_smooth + 0.2 * fps_raw  # EMA
            t_prev  = t_now

            vis = self._draw(frame, results, fps_smooth)
            self._log_frame(frame_idx, latency, fps_smooth, results)

            if writer:
                writer.write(vis)
            if show:
                cv2.imshow("Smart Factory Monitor", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  frame {frame_idx:5d}  |  fps={fps_smooth:5.1f}"
                      f"  |  lat={latency:6.1f}ms")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_idx} frames")
        self._export_results()

    def run_images(self, image_dir: str, show: bool = False):
        """Process a directory of images."""
        img_dir = Path(image_dir)
        paths = sorted(list(img_dir.glob("*.jpg")) +
                       list(img_dir.glob("*.png")) +
                       list(img_dir.glob("*.jpeg")))
        if not paths:
            print(f"[WARN] No images found in {image_dir}")
            return

        print(f"[INFO] Processing {len(paths)} images from {image_dir}")
        for idx, p in enumerate(paths):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            results, latency = self._infer(frame)
            fps = 1000.0 / max(latency, 1e-3)
            vis = self._draw(frame, results, fps)
            self._log_frame(idx, latency, fps, results)

            out_img = self.output_dir / f"det_{p.stem}.jpg"
            cv2.imwrite(str(out_img), vis)
            if show:
                cv2.imshow("Detection", vis)
                cv2.waitKey(500)

            print(f"  [{idx+1:3d}/{len(paths)}] {p.name}  lat={latency:.1f}ms")

        cv2.destroyAllWindows()
        self._export_results()

    # ── Export ─────────────────────────────────────────────────────────────

    def _export_results(self):
        """Export benchmark CSV, JSON summary, and print report."""
        if not self.frame_metrics:
            print("[WARN] No metrics to export.")
            return

        latencies = [m["latency_ms"] for m in self.frame_metrics]
        fpss      = [m["fps"]        for m in self.frame_metrics]
        dets      = [m["n_detections"] for m in self.frame_metrics]

        summary = {
            "session_id":       self.session_id,
            "model":            f"YOLOv8{self.model_size}",
            "conf_threshold":   self.conf,
            "iou_threshold":    self.iou,
            "total_frames":     len(self.frame_metrics),
            "avg_fps":          round(np.mean(fpss), 2),
            "min_fps":          round(np.min(fpss), 2),
            "max_fps":          round(np.max(fpss), 2),
            "avg_latency_ms":   round(np.mean(latencies), 2),
            "p95_latency_ms":   round(np.percentile(latencies, 95), 2),
            "p99_latency_ms":   round(np.percentile(latencies, 99), 2),
            "avg_detections_per_frame": round(np.mean(dets), 2),
            "total_detections": int(np.sum(dets)),
            "class_totals":     dict(self.class_counts),
        }

        # JSON
        json_path = self.output_dir / f"benchmark_{self.session_id}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # CSV (per-frame)
        csv_path = self.output_dir / f"frame_metrics_{self.session_id}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "latency_ms",
                                                    "fps", "n_detections"])
            writer.writeheader()
            for m in self.frame_metrics:
                writer.writerow({k: m[k] for k in ["frame", "latency_ms",
                                                     "fps", "n_detections"]})

        # Console report
        print("\n" + "="*55)
        print("  BENCHMARK REPORT — Smart Factory Detection Pipeline")
        print("="*55)
        print(f"  Model          : YOLOv8{self.model_size}")
        print(f"  Frames         : {summary['total_frames']}")
        print(f"  Avg FPS        : {summary['avg_fps']}")
        print(f"  Avg latency    : {summary['avg_latency_ms']} ms")
        print(f"  P95 latency    : {summary['p95_latency_ms']} ms")
        print(f"  P99 latency    : {summary['p99_latency_ms']} ms")
        print(f"  Total detects  : {summary['total_detections']}")
        print(f"  Top classes    :")
        for cls, cnt in sorted(summary["class_totals"].items(),
                                key=lambda x: -x[1])[:5]:
            print(f"    {cls:<20s} {cnt}")
        print("="*55)
        print(f"  JSON  → {json_path}")
        print(f"  CSV   → {csv_path}")
        print("="*55 + "\n")
        return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Smart Factory Real-Time Object Detection Pipeline")
    p.add_argument("--source", default="0",
                   help="Input: 0=webcam, path to video file, or image dir")
    p.add_argument("--model", default="n", choices=["n","s","m","l","x"],
                   help="YOLOv8 model size (n=nano … x=xlarge)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Confidence threshold (default: 0.35)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="IoU NMS threshold (default: 0.45)")
    p.add_argument("--output", default="results",
                   help="Output directory for results")
    p.add_argument("--no-show", action="store_true",
                   help="Disable live display (headless mode)")
    p.add_argument("--no-save", action="store_true",
                   help="Disable video saving")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Max frames to process (0=all)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = IndustrialDetector(
        model_size=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output,
    )

    src = args.source
    if src.isdigit():
        src = int(src)
        detector.run_video(src, show=not args.no_show,
                           save=not args.no_save,
                           max_frames=args.max_frames)
    elif Path(src).is_dir():
        detector.run_images(src, show=not args.no_show)
    else:
        detector.run_video(src, show=not args.no_show,
                           save=not args.no_save,
                           max_frames=args.max_frames)
