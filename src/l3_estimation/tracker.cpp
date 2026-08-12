#include "l3_estimation/tracker.hpp"

#include "l3_estimation/filter_est/tracker.hpp"
#include "l3_estimation/gtsam_est/tracker.hpp"
#include "l6_telemetry/math.hpp"

#include <memory>
#include <utility>

namespace L3Estimation {

Armor toArmorObservation(const L2Perception::Armor& detection, TimePoint timestamp)
{
  // 这里只搬运检测元数据；三维位姿由本帧的 PnpSolver 补充。
  Armor observation;
  observation.class_id = detection.class_id;
  observation.points = detection.corners;
  observation.center = detection.center;
  observation.confidence = static_cast<double>(detection.confidence);
  observation.area = L6Telemetry::polygonArea(detection.corners);
  observation.timestamp = timestamp;
  return observation;
}

bool estimatorBackendAvailable(EstimatorBackend backend) noexcept
{
  switch (backend) {
    case EstimatorBackend::Gtsam:
      return GtsamEst::available();
    case EstimatorBackend::Filter:
      break;
  }
  return true;
}

std::unique_ptr<ITracker> makeTracker(
  EstimatorBackend backend,
  const L1Sensor::CameraCalibration& calibration,
  ArmorConfig armor_config,
  TrackerConfig tracker_config)
{
  switch (backend) {
    case EstimatorBackend::Gtsam:
      // GTSAM 没编进来时这里会抛，调用方不需要自己先查 available()。
      return std::make_unique<GtsamEst::Tracker>(
        calibration, std::move(armor_config));

    case EstimatorBackend::Filter:
      break;
  }
  return std::make_unique<FilterEst::Tracker>(
    calibration, std::move(armor_config), std::move(tracker_config));
}

}  // namespace L3Estimation
