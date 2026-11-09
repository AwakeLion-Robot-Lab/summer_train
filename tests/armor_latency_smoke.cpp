// Opt-in replay benchmark: video [device] [model] [OpenCV threads].
// Frames are decoded before timing; this measures detection, not camera latency.
#include "runtime/auto_aim_config.hpp"
#include "runtime/armor_detector_factory.hpp"
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc > 4) cv::setNumThreads(std::stoi(argv[4]));
    std::cout << "opencv_threads=" << cv::getNumThreads() << '\n';
    auto config = runtime::loadAutoAimConfig("config/auto_aim.yaml");
    if (argc > 2) config.inference.device = argv[2];
    if (argc > 3) config.inference.model_path = argv[3];
    auto detector = runtime::makeDetector(config, argc > 3);
    cv::VideoCapture video(argc > 1 ? argv[1] : "video/3m_run_mid.avi");
    std::vector<cv::Mat> frames;
    for (int i = 0; i < 120; ++i) {
      cv::Mat frame;
      if (!video.read(frame)) break;
      frames.push_back(frame);
    }
    if (frames.empty()) throw std::runtime_error("No benchmark frames");
    for (int i = 0; i < 30; ++i) detector.detect(frames[i % frames.size()]);
    std::vector<double> total, infer, decode, refine, preprocess;
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&](std::uint32_t value) { hash ^= value; hash *= 1099511628211ULL; };
    std::size_t detections = 0;
    for (int repeat = 0; repeat < 5; ++repeat) {
      for (const auto& frame : frames) {
        auto armors = detector.detect(frame);
        mix(static_cast<std::uint32_t>(armors.size()));
        detections += armors.size();
        for (const auto& armor : armors) {
          mix(armor.class_id); mix(static_cast<unsigned>(armor.color));
          mix(std::bit_cast<std::uint32_t>(armor.confidence));
          for (const auto& p : armor.corners) {
            mix(std::bit_cast<std::uint32_t>(p.x)); mix(std::bit_cast<std::uint32_t>(p.y));
          }
        }
        const auto t = detector.lastTiming();
        preprocess.push_back(t.preprocess); infer.push_back(t.infer);
        decode.push_back(t.decode); refine.push_back(t.refine);
        total.push_back(t.preprocess + t.infer + t.decode + t.refine + t.side_light);
      }
    }
    const auto report = [](const char* name, std::vector<double> values) {
      std::sort(values.begin(), values.end());
      double sum = 0; for (double v : values) sum += v;
      std::cout << name << " mean_ms=" << sum / values.size()
                << " p50_ms=" << values[values.size()/2]
                << " p95_ms=" << values[values.size()*95/100] << '\n';
    };
    report("total", total); report("preprocess", preprocess); report("infer", infer);
    report("decode", decode); report("refine", refine);
    std::cout << "samples=" << total.size() << " detections=" << detections << " hash=" << hash << '\n';
    if (!detections) throw std::runtime_error("No detections: invalid accuracy baseline");
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
