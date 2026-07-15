#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void printShape(const std::vector<std::size_t>& shape)
{
  std::cout << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      std::cout << ", ";
    }
    std::cout << shape[index];
  }
  std::cout << ']';
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const std::filesystem::path model_path = argc >= 2
      ? std::filesystem::path{argv[1]}
      : std::filesystem::path{"model/armor_model/armor.xml"};
    // 第三个可选参数用于同口径比较 CPU/GPU；不传时保持稳定的 CPU 默认值。
    const std::string device = argc >= 4 ? std::string{argv[3]} : std::string{"CPU"};

    cv::Mat image;
    cv::VideoCapture video;
    bool video_mode = false;
    if (argc >= 3) {
      image = cv::imread(argv[2], cv::IMREAD_COLOR);
      if (image.empty()) {
        // imread() 读不了时再按视频打开；这样同一个参数位置可直接传 jpg/png/mp4。
        video.open(argv[2]);
        require(video.isOpened(), "failed to open input image or video: " + std::string{argv[2]});
        require(video.read(image) && !image.empty(), "input video contains no readable frame");
        video_mode = true;
      }
    } else {
      // 没有测试图片时仍使用真实模型推理；黑图只作为相机帧输入，不伪造模型输出。
      image = cv::Mat(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = model_path;
    model_config.device = device;
    // 与 FosuVision 的实际在线推理保持一致：OpenCV 图像按 BGR 输入模型。
    model_config.model_color_order = L2Perception::ModelColorOrder::Bgr;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);

    require(backend->ready(), "OpenVINO backend did not become ready");
    const auto input_spec = backend->inputSpec();
    require(
      input_spec.shape.size() == 4 && input_spec.shape[0] == 1
      && input_spec.shape[3] == 3,
      "OpenVINO host input is not U8 NHWC");

    std::vector<std::size_t> output_shape;

    // 移交后端所有权前先检查一次原始张量契约，便于把模型/后端错误与 Decoder 错误分开定位。
    {
      const auto preprocessed = L2Perception::ImagePreprocessor::run(image, input_spec);
      const auto raw_result = backend->infer(preprocessed.input);
      require(raw_result.outputs.size() == 1, "armor model must produce exactly one output");
      require(raw_result.outputs.front().name == "output", "unexpected armor output name");
      require(raw_result.outputs.front().isConsistent(), "armor output shape/data mismatch");
      output_shape = raw_result.outputs.front().shape;
      require(output_shape == std::vector<std::size_t>({1, 25200, 22}),
              "unexpected armor output shape");
    }

    // 正式帧全部通过任务级 Detector，覆盖 Backend + Preprocessor + Decoder 的实际编排。
    L2Perception::ArmorDetector armor_detector(std::move(backend));
    require(armor_detector.ready(), "ArmorDetector did not accept the loaded backend");
    const auto detect_frame = [&](const cv::Mat& frame) {
      return armor_detector.detect(frame);
    };

    constexpr int warmup_frames = 5;

    for (int frame = 0; frame < warmup_frames; ++frame) {
      (void)detect_frame(image);
    }

    std::size_t processed_frames = 0;
    std::size_t frames_with_detections = 0;
    std::size_t total_detections = 0;
    std::size_t max_detections_per_frame = 0;
    std::array<std::size_t, 3> color_counts{};  // Red、Blue、Unknown。
    std::array<std::size_t, 9> class_counts{};  // G、1、2、3、4、5、O、Bs、Bb。
    double total_l2_ms = 0.0;

    const auto measure_frame = [&](const cv::Mat& frame) {
      const auto begin = std::chrono::steady_clock::now();
      const auto detections = detect_frame(frame);
      const auto end = std::chrono::steady_clock::now();

      total_l2_ms += std::chrono::duration<double, std::milli>(end - begin).count();
      ++processed_frames;
      total_detections += detections.size();
      if (!detections.empty()) {
        ++frames_with_detections;
      }
      max_detections_per_frame = std::max(max_detections_per_frame, detections.size());

      for (const auto& detection : detections) {
        switch (detection.color) {
        case L2Perception::ArmorColor::Red:
          ++color_counts[0];
          break;
        case L2Perception::ArmorColor::Blue:
          ++color_counts[1];
          break;
        case L2Perception::ArmorColor::Unknown:
          ++color_counts[2];
          break;
        }
        if (detection.class_id >= 0
            && detection.class_id < static_cast<int>(class_counts.size())) {
          ++class_counts[static_cast<std::size_t>(detection.class_id)];
        }
      }
    };

    if (video_mode) {
      // 第一帧已被读取用于预热；先正式统计它，再顺序处理视频剩余全部帧。
      measure_frame(image);
      while (video.read(image)) {
        if (!image.empty()) {
          measure_frame(image);
        }
      }
    } else {
      constexpr int measured_frames = 20;
      for (int frame = 0; frame < measured_frames; ++frame) {
        measure_frame(image);
      }
    }

    require(processed_frames != 0, "no frame was processed");
    const double average_ms = total_l2_ms / static_cast<double>(processed_frames);

    std::cout << "OpenVINO armor smoke passed\ndevice " << device << "\ninput  ";
    printShape(input_spec.shape);
    std::cout << " U8 NHWC BGR\noutput ";
    printShape(output_shape);
    std::cout << " FP32 zero-copy\naverage full L2 time: " << average_ms
              << " ms\nprocessed frames: " << processed_frames
              << "\nframes with detections: " << frames_with_detections
              << "\ntotal detections: " << total_detections
              << "\nmax detections per frame: " << max_detections_per_frame
              << "\ncolors: red=" << color_counts[0] << ", blue=" << color_counts[1]
              << ", unknown=" << color_counts[2]
              << "\nclasses [G,1,2,3,4,5,O,Bs,Bb]: ";
    for (std::size_t index = 0; index < class_counts.size(); ++index) {
      if (index != 0) {
        std::cout << ',';
      }
      std::cout << class_counts[index];
    }
    std::cout << '\n';

    if (video_mode) {
      const double fps = video.get(cv::CAP_PROP_FPS);
      const double declared_frames = video.get(cv::CAP_PROP_FRAME_COUNT);
      std::cout << "video metadata: fps=" << fps << ", declared_frames=" << declared_frames << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "OpenVINO armor smoke failed: " << error.what() << '\n';
    return 1;
  }
}
