#pragma once

#include <Eigen/Core>

#include <array>
#include <chrono>

namespace L3Estimation {

// Stable, read-only snapshot consumed by the preserved L4 planner.
// The first eleven entries match TrackedTarget's state layout exactly.
enum StateIndex : int {
  XC = 0,
  VX = 1,
  YC = 2,
  VY = 3,
  ZC = 4,
  VZ = 5,
  YAW = 6,
  YAW_RATE = 7,
  RADIUS = 8,
  RADIUS_OFFSET = 9,
  HEIGHT_OFFSET = 10,
  STATE_DIM = 11
};

using StateVector = Eigen::Matrix<double, STATE_DIM, 1>;
using StateCovariance = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;

struct TargetState {
  int robot_id{-1};
  int armor_count{4};
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double yaw_rate{0.0};
  double radius{0.0};
  double radius_offset{0.0};
  double height_offset{0.0};
  // Three-armor targets use independent offsets for armor 1 and 2. Four-
  // armor targets continue to use height_offset for the odd armor pair.
  std::array<double, 3> three_armor_height_offsets{0.0, 0.0, 0.0};
  StateCovariance covariance{StateCovariance::Identity()};
  std::chrono::steady_clock::time_point timestamp{};
};

}  // namespace L3Estimation
