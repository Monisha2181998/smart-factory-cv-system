/**
 * Smart Factory Object Detection — C++ Inference Engine
 * Author: Monisha Ravi Kumar
 *
 * Loads a YOLOv8 ONNX model via OpenCV DNN module and runs
 * real-time inference on webcam / video / image input.
 *
 * Build:
 *   g++ -std=c++17 -O2 detector.cpp \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -o smart_factory_detector
 *
 * Run:
 *   ./smart_factory_detector --model yolov8n.onnx --source 0
 *   ./smart_factory_detector --model yolov8n.onnx --source factory.mp4
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <sstream>

// ── Configuration ──────────────────────────────────────────────────────────

struct Config {
    std::string model_path  = "yolov8n.onnx";
    std::string source      = "0";
    std::string output_dir  = "results";
    float  conf_threshold   = 0.35f;
    float  nms_threshold    = 0.45f;
    int    input_width      = 640;
    int    input_height     = 640;
    bool   show_window      = true;
    bool   save_video       = true;
    int    max_frames       = 0;      // 0 = unlimited
};

// COCO class names (80 classes)
const std::vector<std::string> CLASS_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

// ── Utility ────────────────────────────────────────────────────────────────

cv::Scalar classColour(int cls_id) {
    // Deterministic per-class colour
    cv::RNG rng(cls_id * 7 + 13);
    return cv::Scalar(rng.uniform(80, 230),
                      rng.uniform(80, 230),
                      rng.uniform(80, 230));
}

std::string formatMs(double ms) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << ms << " ms";
    return ss.str();
}

// ── Detection result ───────────────────────────────────────────────────────

struct Detection {
    int   class_id;
    float confidence;
    cv::Rect box;
};

// ── YOLOv8 post-processing ─────────────────────────────────────────────────

/**
 * YOLOv8 ONNX output shape: [1, 84, 8400]
 * Rows 0-3  : cx, cy, w, h
 * Rows 4-83 : class scores (80 classes)
 */
std::vector<Detection> postprocess(const cv::Mat& output,
                                   float conf_thresh,
                                   float nms_thresh,
                                   int orig_w, int orig_h,
                                   int input_w, int input_h) {
    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       class_ids;

    // output shape: [1, 84, 8400] → reshape to [84, 8400]
    cv::Mat out = output.reshape(1, output.size[1]);  // 84 x 8400

    float scale_x = (float)orig_w / input_w;
    float scale_y = (float)orig_h / input_h;

    for (int i = 0; i < out.cols; ++i) {
        // Find max class score
        cv::Mat class_scores = out.col(i).rowRange(4, out.rows);
        double max_score;
        cv::Point max_loc;
        cv::minMaxLoc(class_scores, nullptr, &max_score, nullptr, &max_loc);

        if ((float)max_score < conf_thresh) continue;

        float cx = out.at<float>(0, i) * scale_x;
        float cy = out.at<float>(1, i) * scale_y;
        float w  = out.at<float>(2, i) * scale_x;
        float h  = out.at<float>(3, i) * scale_y;

        int x1 = std::max(0, (int)(cx - w / 2));
        int y1 = std::max(0, (int)(cy - h / 2));
        int bw = std::min((int)w, orig_w - x1);
        int bh = std::min((int)h, orig_h - y1);

        boxes.push_back(cv::Rect(x1, y1, bw, bh));
        scores.push_back((float)max_score);
        class_ids.push_back(max_loc.y);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh, nms_thresh, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        detections.push_back({class_ids[idx], scores[idx], boxes[idx]});
    }
    return detections;
}

// ── Drawing ────────────────────────────────────────────────────────────────

void drawDetections(cv::Mat& frame,
                    const std::vector<Detection>& dets,
                    double fps, const std::string& model_name) {
    for (const auto& d : dets) {
        auto colour = classColour(d.class_id);
        cv::rectangle(frame, d.box, colour, 2);

        std::string label = CLASS_NAMES[d.class_id] + " "
                          + std::to_string((int)(d.confidence * 100)) + "%";
        int baseline = 0;
        auto sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                   0.55, 1, &baseline);
        cv::rectangle(frame,
                      cv::Point(d.box.x, d.box.y - sz.height - 8),
                      cv::Point(d.box.x + sz.width + 4, d.box.y),
                      colour, cv::FILLED);
        cv::putText(frame, label,
                    cv::Point(d.box.x + 2, d.box.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1);
    }

    // HUD
    std::ostringstream hud;
    hud << "FPS: " << std::fixed << std::setprecision(1) << fps
        << "  |  " << model_name << "  |  dets=" << dets.size();
    cv::rectangle(frame, cv::Point(0, 0),
                  cv::Point((int)hud.str().size() * 9 + 10, 28),
                  cv::Scalar(20, 20, 20), cv::FILLED);
    cv::putText(frame, hud.str(), cv::Point(6, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(0, 255, 120), 1);
}

// ── Benchmark report ───────────────────────────────────────────────────────

void printReport(const std::string& model_name,
                 const std::vector<double>& latencies) {
    if (latencies.empty()) return;

    auto sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    double avg = std::accumulate(sorted.begin(), sorted.end(), 0.0)
               / sorted.size();
    double p95 = sorted[(size_t)(sorted.size() * 0.95)];
    double p99 = sorted[(size_t)(sorted.size() * 0.99)];
    double fps  = 1000.0 / avg;

    std::cout << "\n" << std::string(55, '=') << "\n";
    std::cout << "  BENCHMARK REPORT — C++ Inference Engine\n";
    std::cout << std::string(55, '=') << "\n";
    std::cout << "  Model        : " << model_name       << "\n";
    std::cout << "  Frames       : " << latencies.size() << "\n";
    std::cout << "  Avg FPS      : " << std::fixed << std::setprecision(1) << fps  << "\n";
    std::cout << "  Avg latency  : " << formatMs(avg) << "\n";
    std::cout << "  P95 latency  : " << formatMs(p95) << "\n";
    std::cout << "  P99 latency  : " << formatMs(p99) << "\n";
    std::cout << "  Min latency  : " << formatMs(sorted.front()) << "\n";
    std::cout << "  Max latency  : " << formatMs(sorted.back())  << "\n";
    std::cout << std::string(55, '=') << "\n\n";
}

// ── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Config cfg;

    // Simple CLI parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if      (arg == "--model"  && i+1 < argc) cfg.model_path    = argv[++i];
        else if (arg == "--source" && i+1 < argc) cfg.source        = argv[++i];
        else if (arg == "--conf"   && i+1 < argc) cfg.conf_threshold= std::stof(argv[++i]);
        else if (arg == "--output" && i+1 < argc) cfg.output_dir    = argv[++i];
        else if (arg == "--no-show")               cfg.show_window   = false;
        else if (arg == "--no-save")               cfg.save_video    = false;
        else if (arg == "--max-frames" && i+1 < argc) cfg.max_frames = std::stoi(argv[++i]);
    }

    std::cout << "[INFO] Smart Factory Detection Pipeline (C++ / OpenCV DNN)\n";
    std::cout << "[INFO] Model  : " << cfg.model_path << "\n";
    std::cout << "[INFO] Source : " << cfg.source     << "\n";
    std::cout << "[INFO] Conf   : " << cfg.conf_threshold << "\n\n";

    // Load model
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(cfg.model_path);
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Failed to load model: " << e.what() << "\n";
        std::cerr << "        Export with: yolo export model=yolov8n.pt format=onnx\n";
        return 1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Open source
    cv::VideoCapture cap;
    bool is_webcam = (cfg.source == "0");
    if (is_webcam) cap.open(0);
    else           cap.open(cfg.source);

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open source: " << cfg.source << "\n";
        return 1;
    }

    int orig_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int orig_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double src_fps = cap.get(cv::CAP_PROP_FPS);
    if (src_fps <= 0) src_fps = 30.0;

    std::cout << "[INFO] Source resolution: " << orig_w << "x" << orig_h
              << "  @" << src_fps << " fps\n";

    // Video writer
    cv::VideoWriter writer;
    if (cfg.save_video) {
        std::string out_path = cfg.output_dir + "/output_cpp.mp4";
        writer.open(out_path,
                    cv::VideoWriter::fourcc('m','p','4','v'),
                    src_fps, cv::Size(orig_w, orig_h));
        std::cout << "[INFO] Saving to: " << out_path << "\n";
    }

    std::vector<double> latencies;
    double fps_smooth = 0.0;
    auto t_prev = std::chrono::steady_clock::now();
    int frame_idx = 0;

    std::cout << "[INFO] Running — press Q to quit\n\n";

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;
        if (cfg.max_frames > 0 && frame_idx >= cfg.max_frames) break;

        // Pre-process: letterbox → blob
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                               cv::Size(cfg.input_width, cfg.input_height),
                               cv::Scalar(), true, false);

        // Inference
        auto t0 = std::chrono::steady_clock::now();
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        auto t1 = std::chrono::steady_clock::now();

        double latency = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(latency);

        // FPS (EMA)
        auto t_now = std::chrono::steady_clock::now();
        double raw_fps = 1e3 / std::chrono::duration<double, std::milli>(
                                   t_now - t_prev).count();
        fps_smooth = 0.8 * fps_smooth + 0.2 * raw_fps;
        t_prev = t_now;

        // Post-process
        auto dets = postprocess(outputs[0], cfg.conf_threshold,
                                cfg.nms_threshold,
                                orig_w, orig_h,
                                cfg.input_width, cfg.input_height);

        // Draw
        drawDetections(frame, dets, fps_smooth, "YOLOv8-ONNX");

        if (cfg.save_video && writer.isOpened()) writer.write(frame);
        if (cfg.show_window) {
            cv::imshow("Smart Factory Monitor (C++)", frame);
            if ((cv::waitKey(1) & 0xFF) == 'q') break;
        }

        frame_idx++;
        if (frame_idx % 30 == 0) {
            std::cout << "  frame " << std::setw(5) << frame_idx
                      << "  |  fps=" << std::fixed << std::setprecision(1)
                      << fps_smooth
                      << "  |  lat=" << formatMs(latency)
                      << "  |  dets=" << dets.size() << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();

    printReport(cfg.model_path, latencies);
    return 0;
}
