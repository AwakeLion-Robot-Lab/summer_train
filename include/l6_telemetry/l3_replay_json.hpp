#pragma once

#include "l3_estimation/types.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <nlohmann/json.hpp>

namespace L6Telemetry {

// 为 PlotJuggler 生成固定路径的单帧调试数据。空指针表示本帧没有对应的
// L3 观测或目标；这时只发布 valid=false，不伪造零姿态样本。
nlohmann::json makeL3ReplayJson(
  int frame_index,
  double time_seconds,
  const Eigen::Vector3d& gimbal_rpy_rad,
  const L3Estimation::ArmorObservation* observation,
  const L3Estimation::TargetState* target,
  bool geometry_constraints_enabled,
  double l2_ms,
  double l3_ms,
  double realtime_lag_ms,
  std::size_t skipped_frames);

}  // namespace L6Telemetry
