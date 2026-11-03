#include "l3_planning/in_plate_aim.hpp"
#include "l2_estimation/armor/vehicle_model.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace L4Planning {
namespace {

double wrap(double x) { return std::remainder(x, 2.0 * std::numbers::pi); }

Eigen::Vector2d movementLimits(
  const PredictionResult& prediction, const ArmorPose& armor,
  const InPlateAimConfig& config, const Eigen::Vector3d& gun_origin_world,
  const Eigen::Matrix3d* aligned_vehicle_rotation = nullptr)
{
  const auto& target = prediction.predicted_vehicle;
  const auto& covariance = target.covariance;
  Eigen::Matrix3d position_covariance;
  const int indices[] = {L3Estimation::XC, L3Estimation::YC, L3Estimation::ZC};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      position_covariance(i, j) = covariance(indices[i], indices[j]);
    }
  }
  double shape_variance = 0.0;
  if (target.filter_state) {
    // 使用完整 SO(3) 误差状态协方差，包含倾斜、半径和高度及其交叉项。
    namespace VM = L3Estimation::VehicleModel;
    const auto& filter = *target.filter_state;
    const auto& x = filter.rawState();
    const Eigen::Matrix3d rotation = aligned_vehicle_rotation
      ? *aligned_vehicle_rotation : VM::vehicleRotation<double>(x.data(), filter.name);
    const Eigen::Vector3d local = rotation.transpose() * (armor.position_world - target.center);
    Eigen::Matrix3d skew;
    skew << 0.0, -local.z(), local.y(), local.z(), 0.0, -local.x(), -local.y(), local.x(), 0.0;
    Eigen::Matrix<double, 3, VM::kStateSize> jacobian =
      Eigen::Matrix<double, 3, VM::kStateSize>::Zero();
    jacobian(0, VM::idx::CX) = jacobian(1, VM::idx::CY) = jacobian(2, VM::idx::CZ) = 1.0;
    const Eigen::Matrix3d rotation_jacobian = -rotation * skew;
    // 前哨/基地的几何模型忽略 roll/pitch；这些不可观测方差不能压缩板内范围。
    if (!VM::yawOnlyTarget(filter.name)) {
      jacobian.col(VM::idx::ROT_X) = rotation_jacobian.col(0);
      jacobian.col(VM::idx::ROT_Y) = rotation_jacobian.col(1);
    }
    jacobian.col(VM::idx::ROT_Z) = rotation_jacobian.col(2);
    const bool second = target.armor_count == 4 && armor.armor_id % 2 != 0;
    jacobian.col(second ? VM::idx::LOG_R2 : VM::idx::LOG_R1) =
      rotation * Eigen::Vector3d{local.x(), local.y(), 0.0};
    if (second) jacobian.col(VM::idx::HEIGHT) = rotation.col(2);
    if (filter.name == L3Estimation::ArmorName::Outpost && armor.armor_id > 0) {
      jacobian.col(armor.armor_id == 1 ? VM::idx::OUTPOST_DZ1 : VM::idx::OUTPOST_DZ2) = rotation.col(2);
    }
    position_covariance = jacobian * filter.covariance() * jacobian.transpose();
  } else {
    const double radius = (armor.position_world - target.center).norm();
    // 数值兼容入口：保守覆盖未展开的半径、角度和高度差误差。
    shape_variance = std::max(0.0, covariance(L3Estimation::RADIUS, L3Estimation::RADIUS))
      + std::max(0.0, covariance(L3Estimation::RADIUS_OFFSET, L3Estimation::RADIUS_OFFSET))
      + radius * radius * std::max(0.0, covariance(L3Estimation::YAW, L3Estimation::YAW))
      + std::max(0.0, covariance(L3Estimation::HEIGHT_OFFSET, L3Estimation::HEIGHT_OFFSET));
  }
  const Eigen::Vector3d sight = armor.position_world - gun_origin_world;
  const double facing = sight.norm() > 1e-6
    ? std::clamp(armor.rotation_world.col(0).dot(sight.normalized()), 0.0, 1.0) : 0.0;
  Eigen::Vector2d limits;
  const double width = armor.armor_type == ArmorType::Large
    ? config.large_width : config.small_width;
  for (int axis = 0; axis < 2; ++axis) {
    const Eigen::Vector3d direction = armor.rotation_world.col(axis + 1);
    const double variance = direction.dot(position_covariance * direction) + shape_variance;
    const double sigma = std::sqrt(std::max(0.0, variance));
    const double half_size = 0.5 * (axis == 0 ? width : config.height) * config.hit_margin_ratio;
    limits[axis] = std::max(0.0, half_size - config.edge_margin - config.uncertainty_sigma * sigma) * facing;
  }
  return limits;
}

}  // namespace

int nextArmorInRotation(int armor_id, int armor_count, int direction) noexcept
{
  if ((armor_count != 3 && armor_count != 4) || armor_id < 0
      || armor_id >= armor_count || (direction != 1 && direction != -1)) {
    return -1;
  }
  return (armor_id - direction + armor_count) % armor_count;
}

bool validInPlateAimConfig(const InPlateAimConfig& c) noexcept
{
  const auto positive = [](double x) { return std::isfinite(x) && x > 0.0; };
  const auto nonnegative = [](double x) { return std::isfinite(x) && x >= 0.0; };
  return positive(c.small_width) && positive(c.large_width) && positive(c.height)
    && positive(c.hit_margin_ratio) && c.hit_margin_ratio <= 1.0
    && nonnegative(c.edge_margin)
    && c.edge_margin < 0.5 * std::min({c.small_width, c.large_width, c.height}) * c.hit_margin_ratio
    && nonnegative(c.uncertainty_sigma) && nonnegative(c.lookahead)
    && positive(c.max_speed) && positive(c.max_acceleration) && positive(c.response_time)
    && nonnegative(c.direction_exit_rate) && positive(c.direction_enter_rate)
    && c.direction_enter_rate > c.direction_exit_rate && c.warmup_frames > 0
    && (c.mode == InPlateAimMode::Center || c.mode == InPlateAimMode::FixedDiagonal
        || c.mode == InPlateAimMode::Independent);
}

InPlateAimPoint queryInPlateAim(
  const PredictionResult& prediction, const ArmorPose& armor,
  const InPlateAimState& state, TimePoint control_time, double gun_yaw,
  const InPlateAimConfig& config, const Eigen::Vector3d& gun_origin_world)
{
  InPlateAimPoint result;
  result.position_world = armor.position_world;
  if (!prediction.valid || !armor.valid || !armor.rotation_world.allFinite()
      || !std::isfinite(gun_yaw) || !gun_origin_world.allFinite() || !validInPlateAimConfig(config)) {
    return result;
  }
  result.valid = true;
  result.limits = movementLimits(prediction, armor, config, gun_origin_world);
  const auto& vehicle = prediction.predicted_vehicle;
  const Eigen::Vector3d center_sight = vehicle.center - gun_origin_world;
  const double distance_squared = center_sight.head<2>().squaredNorm();
  const double bearing_rate = distance_squared > 1e-9
    ? (center_sight.x() * vehicle.velocity.y() - center_sight.y() * vehicle.velocity.x()) / distance_squared : 0.0;
  const double rate = vehicle.yaw_rate - bearing_rate;
  int direction = state.robot_id == armor.robot_id ? state.direction : 0;
  if (std::abs(rate) >= config.direction_enter_rate) {
    direction = rate > 0.0 ? 1 : -1;
  } else if (std::abs(rate) <= config.direction_exit_rate || direction * rate < 0.0) {
    // 反向尚未确认时先停止方向预调，不继续朝旧方向寻找接替板。
    direction = 0;
  }
  result.direction = direction;
  if (!config.enable || config.mode == InPlateAimMode::Center
      || state.robot_id != armor.robot_id || state.stable_frames < config.warmup_frames) {
    return result;
  }

  Eigen::Vector2d desired = Eigen::Vector2d::Zero();
  if (direction != 0) {
    const Eigen::Vector3d sight = armor.position_world - gun_origin_world;
    const double range_squared = sight.head<2>().squaredNorm();
    const Eigen::Vector3d u = armor.rotation_world.col(1);
    const double yaw_per_u = range_squared > 1e-9
      ? (sight.x() * u.y() - sight.y() * u.x()) / range_squared : 0.0;
    if (std::abs(yaw_per_u) > 1e-5) {
      const double bearing = std::atan2(sight.y(), sight.x());
      const double plate_bearing_rate = range_squared > 1e-9
        ? (sight.x() * armor.velocity_world.y() - sight.y() * armor.velocity_world.x()) / range_squared : 0.0;
      // 沿当前枪口速度延伸，补偿未来短时板心相对视线的运动。
      desired.x() = (wrap(gun_yaw - bearing)
        + (state.gun_yaw_rate - plate_bearing_rate) * config.lookahead) / yaw_per_u;
    }

    // 板号由整车几何确定，使用循环顺序，三板步进 120°、四板步进 90°。
    const ArmorPose* next = nullptr;
    result.next_armor_id = nextArmorInRotation(armor.armor_id, vehicle.armor_count, direction);
    const double next_phase = 2.0 * std::numbers::pi / vehicle.armor_count;
    for (const auto& other : prediction.armor_candidates) {
      if (other.valid && other.armor_id == result.next_armor_id) {
        next = &other;
        break;
      }
    }
    desired.x() = std::clamp(desired.x(), -result.limits.x(), result.limits.x());
    const double vertical_axis = armor.rotation_world(2, 2);
    const double base_height = armor.position_world.z() + desired.x() * armor.rotation_world(2, 1);
    if (next && std::abs(vertical_axis) > 0.1) {
      // 在相同面向角比较高度，不依赖转速或任意的外推时间截断。
      // 绕车体自身 z 轴转相邻板间角，保留真实半径、高度和车体倾斜。
      const auto height_interval = [&](const ArmorPose& other) {
        const Eigen::Matrix3d vehicle_rotation = vehicle.filter_state
          ? L3Estimation::VehicleModel::vehicleRotation<double>(
              vehicle.filter_state->rawState().data(), vehicle.filter_state->name)
          : L3Estimation::VehicleModel::rotationZY(vehicle.yaw, 0.0);
        const double delta = wrap((armor.armor_id - other.armor_id) * next_phase);
        const Eigen::Matrix3d aligned_vehicle_rotation = vehicle_rotation
          * Eigen::AngleAxisd(delta, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        const Eigen::Matrix3d turn = aligned_vehicle_rotation * vehicle_rotation.transpose();
        ArmorPose aligned = other;
        aligned.position_world = vehicle.center + turn * (other.position_world - vehicle.center);
        aligned.rotation_world = turn * other.rotation_world;
        const auto limits = movementLimits(prediction, aligned, config, gun_origin_world,
          &aligned_vehicle_rotation);
        const double reach = std::abs(aligned.rotation_world(2, 2)) * limits.y();
        return Eigen::Vector2d{aligned.position_world.z() - reach, aligned.position_world.z() + reach};
      };
      const double reach = std::abs(vertical_axis) * result.limits.y();
      const double current_low = base_height - reach;
      const double current_high = base_height + reach;
      const auto next_interval = height_interval(*next);
      double low = std::max(current_low, next_interval.x());
      double high = std::min(current_high, next_interval.y());
      const bool outpost = vehicle.armor_count == 3
        && vehicle.robot_id == static_cast<int>(L3Estimation::ArmorName::Outpost);
      if (outpost && config.outpost_common_height && config.mode == InPlateAimMode::Independent) {
        double common_low = current_low;
        double common_high = current_high;
        for (const auto& other : prediction.armor_candidates) {
          if (!other.valid) continue;
          const auto interval = height_interval(other);
          common_low = std::max(common_low, interval.x());
          common_high = std::min(common_high, interval.y());
        }
        if (common_low <= common_high) {
          low = common_low;
          high = common_high;
          result.using_common_height = true;
        }
      }
      const double next_height = 0.5 * (next_interval.x() + next_interval.y());
      const double height = low <= high ? 0.5 * (low + high) : next_height;
      desired.y() = (height - base_height) / vertical_axis;
    }
  }
  for (int axis = 0; axis < 2; ++axis) {
    desired[axis] = std::clamp(desired[axis], -result.limits[axis], result.limits[axis]);
  }
  if (config.mode == InPlateAimMode::FixedDiagonal && result.limits.x() > 1e-9) {
    desired.y() = desired.x() / result.limits.x() * result.limits.y();
  }

  const bool same_plate = state.armor_id == armor.armor_id;
  Eigen::Vector2d offset = same_plate ? state.offset : Eigen::Vector2d::Zero();
  Eigen::Vector2d velocity = same_plate ? state.velocity : Eigen::Vector2d::Zero();
  if (!same_plate && state.armor_id >= 0 && std::abs(armor.rotation_world(2, 2)) > 0.1) {
    const auto previous = std::find_if(prediction.armor_candidates.begin(), prediction.armor_candidates.end(),
      [&state](const ArmorPose& other) { return other.valid && other.armor_id == state.armor_id; });
    if (previous != prediction.armor_candidates.end()) {
      // 把上一板的局部偏移带到本次命中时刻，避免冻结上一帧世界高度。
      const Eigen::Vector3d previous_point = previous->position_world
        + previous->rotation_world.col(1) * state.offset.x()
        + previous->rotation_world.col(2) * state.offset.y();
      offset.y() = std::clamp((previous_point.z() - armor.position_world.z())
        / armor.rotation_world(2, 2), -result.limits.y(), result.limits.y());
      const double mapped_velocity = (previous->velocity_world.z() - armor.velocity_world.z()
        + previous->rotation_world(2, 1) * state.velocity.x()
        + previous->rotation_world(2, 2) * state.velocity.y()) / armor.rotation_world(2, 2);
      velocity.y() = std::clamp(mapped_velocity, -config.max_speed, config.max_speed);
    }
  }
  // 使用控制时刻推进，弹道的每次飞行时间假设不会反复推进状态。
  double remaining = std::clamp(std::chrono::duration<double>(control_time - state.timestamp).count(), 0.0, 1.0);
  const auto advance = [&](double& position, double& speed, double goal,
                           double max_speed, double max_acceleration, double dt) {
    const double error = goal - position;
    const double speed_bound = std::min(max_speed, std::sqrt(2.0 * max_acceleration * std::abs(error)));
    const double target_velocity = std::clamp(error / config.response_time, -speed_bound, speed_bound);
    const double acceleration = std::clamp((target_velocity - speed) / dt,
      -max_acceleration, max_acceleration);
    position += speed * dt + 0.5 * acceleration * dt * dt;
    speed += acceleration * dt;
  };
  if (config.mode == InPlateAimMode::FixedDiagonal && result.limits.minCoeff() > 1e-9) {
    // 固定对角线使用单一进度，整个过渡始终位于该直线上。
    double progress = offset.x() / result.limits.x();
    double speed = velocity.x() / result.limits.x();
    const double goal = desired.x() / result.limits.x();
    const double scale = result.limits.maxCoeff();
    while (remaining > 1e-9) {
      const double dt = std::min(0.005, remaining);
      advance(progress, speed, goal, config.max_speed / scale, config.max_acceleration / scale, dt);
      remaining -= dt;
    }
    offset = result.limits * progress;
    velocity = result.limits * speed;
  } else {
    while (remaining > 1e-9) {
      const double dt = std::min(0.005, remaining);
      for (int axis = 0; axis < 2; ++axis) {
        advance(offset[axis], velocity[axis], desired[axis], config.max_speed, config.max_acceleration, dt);
      }
      remaining -= dt;
    }
  }
  // 安全区域骤缩时优先留在有效板面；此时允许打断运动约束。
  for (int axis = 0; axis < 2; ++axis) {
    const double bounded = std::clamp(offset[axis], -result.limits[axis], result.limits[axis]);
    if (bounded != offset[axis]) velocity[axis] = 0.0;
    offset[axis] = bounded;
  }
  result.offset = offset;
  result.velocity = velocity;
  result.position_world += armor.rotation_world.col(1) * offset.x()
    + armor.rotation_world.col(2) * offset.y();
  return result;
}

}  // namespace L4Planning
