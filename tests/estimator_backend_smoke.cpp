// 估计器后端切换的冒烟测试。验证的是**接缝**本身，不是某个估计器的精度：
//   - 名字 <-> 枚举的往返转换；
//   - makeTracker() 按枚举给出对应后端，且返回的对象满足 ITracker 的基本契约；
//   - 请求一个没编进来的后端时抛异常，而不是悄悄回退到 EKF。
// 后一条是这个文件存在的主要理由：静默回退会让回放曲线看起来是因子图跑出来的。
#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

int failure_count = 0;

void expect(bool condition, std::string_view message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failure_count;
  }
}

}  // namespace

int main()
{
  using L3Estimation::EstimatorBackend;

  // 名字往返。YAML 里写错一个字母不该被当成合法后端。
  expect(
    L3Estimation::estimatorBackendFromString("filter") == EstimatorBackend::Filter &&
      L3Estimation::estimatorBackendFromString("ekf") == EstimatorBackend::Filter &&
      L3Estimation::estimatorBackendFromString("gtsam") == EstimatorBackend::Gtsam &&
      L3Estimation::estimatorBackendFromString("fgo") == EstimatorBackend::Gtsam,
    "estimatorBackendFromString does not accept the documented names");
  expect(
    !L3Estimation::estimatorBackendFromString("kalman").has_value() &&
      !L3Estimation::estimatorBackendFromString("").has_value(),
    "estimatorBackendFromString accepts an unknown backend name");
  expect(
    L3Estimation::toString(EstimatorBackend::Filter) == "filter" &&
      L3Estimation::toString(EstimatorBackend::Gtsam) == "gtsam",
    "toString does not round-trip with estimatorBackendFromString");

  // filter 后端在任何构建里都必须可用。
  expect(
    L3Estimation::estimatorBackendAvailable(EstimatorBackend::Filter),
    "the filter backend must always be available");

  const auto camera_config = YAML::LoadFile("tests/data/camera_calibration_inline.yaml");
  const auto calibration = L1Sensor::loadCameraCalibration(
    camera_config["calibration"], "Estimator backend smoke config");

  const auto filter_tracker = L3Estimation::makeTracker(EstimatorBackend::Filter, calibration);
  expect(filter_tracker != nullptr, "makeTracker returned null for the filter backend");
  if (filter_tracker) {
    expect(
      filter_tracker->backend() == EstimatorBackend::Filter,
      "makeTracker(Filter) produced a different backend");
    expect(filter_tracker->ready(), "the filter tracker rejected a valid calibration");
    expect(
      filter_tracker->state() == L3Estimation::TrackState::Lost &&
        filter_tracker->observations().empty() &&
        filter_tracker->targetArmorPoses().empty(),
      "a fresh tracker must start Lost with no observations");
  }

  // gtsam 后端：编进来了就要能构造并自报家门；没编进来就必须抛异常。
  if (L3Estimation::estimatorBackendAvailable(EstimatorBackend::Gtsam)) {
    const auto gtsam_tracker = L3Estimation::makeTracker(EstimatorBackend::Gtsam, calibration);
    expect(gtsam_tracker != nullptr, "makeTracker returned null for the gtsam backend");
    if (gtsam_tracker) {
      expect(
        gtsam_tracker->backend() == EstimatorBackend::Gtsam,
        "makeTracker(Gtsam) produced a different backend");
    }
  } else {
    bool threw = false;
    try {
      const auto gtsam_tracker = L3Estimation::makeTracker(EstimatorBackend::Gtsam, calibration);
      (void)gtsam_tracker;
    } catch (const std::runtime_error&) {
      threw = true;
    }
    expect(threw, "makeTracker(Gtsam) must throw when GTSAM is not compiled in");
  }

  if (failure_count > 0) {
    std::cerr << "estimator backend smoke test failed with " << failure_count
              << " error(s)\n";
    return 1;
  }
  std::cout << "estimator backend smoke test passed\n";
  return 0;
}
