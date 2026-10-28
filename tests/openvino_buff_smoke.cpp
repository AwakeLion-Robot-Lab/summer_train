#include "l2_perception/buff/buff_detector.hpp"
#include "runtime/buff/buff_config.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message)
{
  if (!ok) throw std::runtime_error(message);
}
}

// 可选参数：一张同时含红色待打靶和中心 R 的原始图，额外验证真实检测和颜色。
int main(int argc, char** argv)
{
  try {
    auto config = runtime::buff::loadConfig("config/buff.yaml").detector;
    config.use_model = true;
    config.model.device = "CPU";
    L2Perception::Buff::Detector detector(config);
    const auto calibration = L1Sensor::loadCameraCalibration(
      YAML::LoadFile("config/camera_config.yaml")["calibration"], "config/camera_config.yaml");
    detector.setCalibration(calibration);
    detector.setEnemyColor(L1Sensor::EnemyColor::Red);
    const cv::Mat black = cv::Mat::zeros(calibration.image_size, CV_8UC3);
    for (int i = 0; i < 2; ++i) {
      const auto result = detector.detect(black);
      require(result.model_detections.empty() && result.bullseyes.empty(),
              "black frames must not create targets");
    }
    if (argc > 1) {
      const auto image = cv::imread(argv[1]);
      require(!image.empty(), "cannot read sample image");
      const auto red = detector.detect(image);
      require(!red.model_detections.empty(), "red sample must produce model detections");
      require(!red.icons.empty() && !red.bullseyes.empty(), "model ROI must reach typed geometry");
      bool target = false;
      for (const auto& bullseye : red.bullseyes) target |= !bullseye.active;
      require(target, "red sample must contain a refined inactive target");
      detector.setEnemyColor(L1Sensor::EnemyColor::Blue);
      require(detector.detect(image).model_detections.empty(), "red frame must be rejected for blue enemy");
      cv::Mat swapped;
      cv::cvtColor(image, swapped, cv::COLOR_BGR2RGB);
      const auto blue = detector.detect(swapped);
      require(!blue.model_detections.empty(), "channel-swapped sample must be detected as blue");
      detector.setEnemyColor(L1Sensor::EnemyColor::Unknown);
      require(detector.detect(image).model_detections.empty(), "unknown enemy must not detect");
      std::cout << "red model=" << red.model_detections.size() << " icons=" << red.icons.size()
                << " refined=" << red.bullseyes.size() << " blue=" << blue.model_detections.size() << '\n';
    }
    std::cout << "OpenVINO Buff smoke test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
