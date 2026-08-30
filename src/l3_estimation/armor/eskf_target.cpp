#include "l3_estimation/armor/eskf_target.hpp"

#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace L3Estimation {

namespace VM = VehicleModel;

namespace {

// 装甲板四角在图像上的顺序：左上、右上、右下、左下。预测侧由左右灯条的上下
// 端点拼出同一顺序，两边必须一致，否则四边形代价算的是两个不同形状。
constexpr int kCorners = 4;

double initialRadiusFor(ArmorName name, const EskfTargetConfig & config)
{
  if (name == ArmorName::Outpost) {
    return config.initial_radius_outpost;
  }
  if (name == ArmorName::BaseSmall || name == ArmorName::BaseLarge) {
    return config.initial_radius_base;
  }
  return config.initial_radius;
}

// 板朝向相机的程度。板的 x 轴指向车心，所以朝外的法向是 -axis_x；它与
// "板 → 相机"方向的点积越大，板越正对。
double facingScore(const Eigen::Isometry3d & armor_in_camera)
{
  const Eigen::Vector3d front_normal = -armor_in_camera.linear().col(0);
  return front_normal.dot(-armor_in_camera.translation());
}

double segmentAngle(const cv::Point2f & from, const cv::Point2f & to)
{
  return std::atan2(to.y - from.y, to.x - from.x);
}

}  // namespace

void EskfTarget::reset(
  const Armor & armor, const EskfTargetConfig & config, TimePoint timestamp,
  const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world)
{
  (void)calibration;
  (void)camera_in_world;

  config_ = config;
  armor_config_ = config.armor;
  name = armor.name;
  armor_type = armor.type;

  const double radius = initialRadiusFor(name, config_);

  // 初始协方差全靠先验给量级。角速度最不确定——单帧完全看不出车在不在转，
  // 给 100 等于告诉滤波器"这一维我基本不知道，请大胆用观测改它"。
  Eigen::Matrix<double, VM::kStateSize, VM::kStateSize> p0;
  p0.setZero();
  p0.diagonal()[VM::idx::CX] = p0.diagonal()[VM::idx::CY] = p0.diagonal()[VM::idx::CZ] = 1.0;
  p0.diagonal()[VM::idx::VCX] = p0.diagonal()[VM::idx::VCY] = p0.diagonal()[VM::idx::VCZ] =
    10.0;
  p0.diagonal()[VM::idx::ROT_X] = p0.diagonal()[VM::idx::ROT_Y] =
    p0.diagonal()[VM::idx::ROT_Z] = 1.0;
  p0.diagonal()[VM::idx::LOG_R1] = p0.diagonal()[VM::idx::P1] =
    p0.diagonal()[VM::idx::P2] = 1.0;
  p0.diagonal()[VM::idx::VYAW] = 100.0;

  // 由一块板的位姿反推整车位姿。T_car = T_armor · (T_armor^car)⁻¹。
  //
  // T_armor^car 依赖两个未知量：板编号和半径，这里都用假设值——**假设看到的
  // 是 0 号板**（θ=0，于是板心在车体系的 (-r, 0, 0)），半径取经验先验。
  //
  // 两个假设都会错但都不致命：编号只是标签的循环平移，认错了整车 yaw 差
  // 2πk/N 而几何仍然自洽；半径偏差由后续观测修正（p0 给了足够不确定性）。
  Eigen::Isometry3d armor_in_world = Eigen::Isometry3d::Identity();
  armor_in_world.translation() = armor.xyz_in_world;
  armor_in_world.linear() = L6Telemetry::yprToRotation(armor.ypr_in_world);

  Eigen::Isometry3d armor_in_vehicle = Eigen::Isometry3d::Identity();
  armor_in_vehicle.translation() = Eigen::Vector3d(-radius, 0.0, 0.0);
  armor_in_vehicle.linear() =
    VM::rotationZY<double>(0.0, armorPitchOf(name));

  const Eigen::Isometry3d vehicle_in_world = armor_in_world * armor_in_vehicle.inverse();

  x_.setZero();
  x_[VM::idx::CX] = vehicle_in_world.translation().x();
  x_[VM::idx::CY] = vehicle_in_world.translation().y();
  x_[VM::idx::CZ] = vehicle_in_world.translation().z();
  x_[VM::idx::LOG_R1] = std::log(radius);
  if (name != ArmorName::Outpost) {
    x_[VM::idx::LOG_R2] = std::log(radius);
  }
  const Eigen::Vector3d rotation = so3Log<double>(vehicle_in_world.linear());
  x_[VM::idx::ROT_X] = rotation.x();
  x_[VM::idx::ROT_Y] = rotation.y();
  x_[VM::idx::ROT_Z] = rotation.z();
  // 速度、角速度、高度差全部从零起步。

  const auto inject = [](const auto & delta, auto & nominal) {
    VM::injectState(delta, nominal);
  };
  const auto box_minus = [](const auto & nominal, const auto & value, auto & delta) {
    VM::boxMinusState(nominal, value, delta);
  };
  const auto zero_q = []() {
    return Eigen::Matrix<double, VM::kStateSize, VM::kStateSize>::Zero();
  };

  filter_.emplace(
    VM::Motion{.dt = 0.005, .name = name}, zero_q, inject, box_minus, p0);
  filter_->setState(x_);
  filter_->setIterationNum(config_.iteration_num);

  t_ = timestamp;
  initialized_ = true;
  converged_ = false;
  jumped = false;
  last_id = 0;
  update_count_ = 0;
}

void EskfTarget::predictEkf(TimePoint timestamp)
{
  if (!filter_) {
    return;
  }
  const double dt = std::chrono::duration<double>(timestamp - t_).count();

  filter_->setPredictFunc(VM::Motion{.dt = dt, .name = name});
  // Q 依赖当前姿态（要旋到世界系）和当前半径（log 换算），必须在推进前按当前
  // 状态求值，所以传的是 lambda 而不是值。
  filter_->setUpdateQ([this, dt]() {
    return VM::processNoise(x_, dt, name, config_.noise);
  });

  x_ = filter_->predict();
  t_ = timestamp;
}

UvlContext EskfTarget::makeContext(
  int id, bool is_left, const L1Sensor::CameraCalibration & calibration,
  const Eigen::Isometry3d & camera_in_world) const
{
  UvlContext ctx;
  ctx.armor_num = armor_num();
  ctx.id = id;
  ctx.is_left = is_left;
  ctx.name = name;
  ctx.armor_config = armor_config_;
  ctx.camera_in_world = camera_in_world;
  ctx.camera_matrix = calibration.camera_matrix;
  ctx.distortion_coefficients = calibration.distortion_coefficients;
  return ctx;
}

std::pair<cv::Point2f, cv::Point2f> EskfTarget::predictLight(
  int id, bool is_left, const Eigen::VectorXd & state,
  const L1Sensor::CameraCalibration & calibration,
  const Eigen::Isometry3d & camera_in_world) const
{
  const UvlMeasure measure{makeContext(id, is_left, calibration, camera_in_world)};
  return measure.projectedPoints(state);
}

std::vector<std::pair<int, Armor>> EskfTarget::matchArmor(
  const std::vector<Armor> & armors, TimePoint timestamp,
  const L1Sensor::CameraCalibration & calibration,
  const Eigen::Isometry3d & camera_in_world) const
{
  std::vector<std::pair<int, Armor>> result;
  if (armors.empty() || !initialized_) {
    return result;
  }

  constexpr double kMaxCost = 1e9;
  const int count = armor_num();

  // 把状态外推到本帧曝光时刻再做关联。
  EskfTarget predicted = snapshot();
  predicted.predict(timestamp);
  const Eigen::VectorXd state = predicted.x_;

  // 可见性筛选：按"朝向相机的程度"排序，只留最正对的前三块。一辆四板车最多
  // 同时看到两块半，取三是留了余量。这是几何近似而非遮挡模型——它只判板朝不
  // 朝着你，不判板有没有被车身自己挡住。
  std::vector<std::pair<double, int>> facing;
  facing.reserve(count);
  for (int id = 0; id < count; ++id) {
    const auto pose_in_world = VM::armorPose<double>(state.data(), id, count, name);
    const Eigen::Isometry3d pose_in_camera = camera_in_world.inverse() * pose_in_world;
    facing.emplace_back(facingScore(pose_in_camera), id);
  }
  std::sort(facing.begin(), facing.end(), [](const auto & a, const auto & b) {
    return a.first > b.first;
  });

  std::vector<int> candidates;
  const std::size_t visible_count = std::min<std::size_t>(3, facing.size());
  for (std::size_t i = 0; i < visible_count; ++i) {
    candidates.push_back(facing[i].second);
  }

  const int observation_count = static_cast<int>(armors.size());
  const double gate = jumped ? config_.match_gate : config_.match_gate_not_all_init;

  std::vector<std::vector<double>> cost(
    observation_count, std::vector<double>(candidates.size(), kMaxCost + 1.0));

  for (int j = 0; j < observation_count; ++j) {
    const std::array<cv::Point2f, kCorners> & measured = armors[j].points;

    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const int id = candidates[i];
      const auto left = predicted.predictLight(id, true, state, calibration, camera_in_world);
      const auto right =
        predicted.predictLight(id, false, state, calibration, camera_in_world);

      // 预测四边形，顺序与检测角点一致：左上、右上、右下、左下。
      const std::array<cv::Point2f, kCorners> predicted_corners{
        left.first, right.first, right.second, left.second};

      // 抽象成"四边形与四边形匹配"，三项代价各自描述一个自由度：
      //   中心误差 —— 整体位置
      //   边角度误差 —— 姿态
      //   周长比例误差 —— 距离缩放
      cv::Point2f predicted_center(0.0F, 0.0F);
      cv::Point2f measured_center(0.0F, 0.0F);
      for (int k = 0; k < kCorners; ++k) {
        predicted_center += predicted_corners[k];
        measured_center += measured[k];
      }
      predicted_center *= 0.25F;
      measured_center *= 0.25F;
      const double center_error = cv::norm(predicted_center - measured_center);

      double angle_error = 0.0;
      double predicted_perimeter = 0.0;
      double measured_perimeter = 0.0;
      for (int k = 0; k < kCorners; ++k) {
        const int next = (k + 1) % kCorners;
        angle_error += std::abs(L6Telemetry::limit_rad(
          segmentAngle(predicted_corners[k], predicted_corners[next]) -
          segmentAngle(measured[k], measured[next])));
        predicted_perimeter += cv::norm(predicted_corners[k] - predicted_corners[next]);
        measured_perimeter += cv::norm(measured[k] - measured[next]);
      }
      const double side_length_error =
        predicted_perimeter > 1e-6
          ? std::abs(predicted_perimeter - measured_perimeter) / predicted_perimeter
          : kMaxCost;

      const double total = config_.weight_center_error * center_error +
                           config_.weight_angle_error * angle_error +
                           config_.weight_side_length_error * side_length_error;

      if (std::isfinite(total) && total < gate) {
        cost[j][i] = total;
      }
    }
  }

  for (const auto & [observation, candidate] :
       greedyMatch(cost, observation_count, static_cast<int>(candidates.size()), kMaxCost)) {
    result.emplace_back(candidates[candidate], armors[observation]);
  }
  return result;
}

int EskfTarget::update(
  const std::vector<std::pair<int, Armor>> & matched, TimePoint timestamp,
  const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world)
{
  if (matched.empty() || !filter_) {
    return 0;
  }

  std::vector<std::shared_ptr<Filter::ObsBase>> observations;

  // 一条灯条 → 一个四维观测。
  const auto addLight = [&](const cv::Point2f & top, const cv::Point2f & bottom, int id,
                            bool is_left) {
    const UvlMeasure measure{makeContext(id, is_left, calibration, camera_in_world)};
    const UvlVector z = uvlMeasurementFrom(top, bottom);

    // R 按灯条像素长度缩放。除以 2 是因为一块板拆成两条灯条、信息量翻倍。
    const double length = cv::norm(top - bottom);
    const double sigma_pixel = config_.sigma_pixel_by_length * length;
    const double sigma_length = config_.sigma_length_by_length * length;
    const double sigma_angle = config_.sigma_angle;

    Eigen::Matrix<double, kUvlMeasureSize, kUvlMeasureSize> r_cov;
    r_cov.setZero();
    r_cov(uvl::ANGLE, uvl::ANGLE) = sigma_angle * sigma_angle / 2.0;
    r_cov(uvl::CENTER_X, uvl::CENTER_X) = sigma_pixel * sigma_pixel / 2.0;
    r_cov(uvl::CENTER_Y, uvl::CENTER_Y) = sigma_pixel * sigma_pixel / 2.0;
    r_cov(uvl::LENGTH, uvl::LENGTH) = sigma_length * sigma_length / 2.0;

    observations.push_back(Filter::makeObs<kUvlMeasureSize>(
      z, measure, [r_cov](const UvlVector &) { return r_cov; },
      [](const UvlVector & z_pred, const UvlVector & z_obs) {
        return UvlMeasure::residual<double>(z_pred, z_obs);
      }));
  };

  for (const auto & [id, armor] : matched) {
    // 见过 0 号以外的板 —— 粘滞置位，此后整车 yaw 与第二组半径才真正可观测。
    jumped = jumped || (id != 0);
    last_id = id;

    // 一块完整板拆成左右两条灯条。角点序左上、右上、右下、左下：
    // 左灯条取 [0]、[3]，右灯条取 [1]、[2]。
    addLight(armor.points[0], armor.points[3], id, true);
    addLight(armor.points[1], armor.points[2], id, false);
  }

  if (observations.empty()) {
    return 0;
  }

  x_ = filter_->updateMulti(observations);
  t_ = timestamp;
  update_count_ += static_cast<int>(observations.size());
  if (update_count_ > 20 && !diverged()) {
    converged_ = true;
  }
  return static_cast<int>(observations.size());
}

Eigen::VectorXd EskfTarget::ekf_x() const
{
  Eigen::VectorXd out = x_;
  // 对外吐线性半径：内部的 log 表示不能泄漏给 L4。
  out[VM::idx::LOG_R1] = std::exp(x_[VM::idx::LOG_R1]);
  if (name != ArmorName::Outpost) {
    out[VM::idx::LOG_R2] = std::exp(x_[VM::idx::LOG_R2]);
  }
  return out;
}

int EskfTarget::armor_num() const noexcept
{
  return armorCountOf(name).value_or(4);
}

void EskfTarget::predict(double dt)
{
  const VM::Motion motion{.dt = dt, .name = name};
  State next;
  motion(x_.data(), next.data());
  x_ = next;
  t_ += std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(dt));
}

void EskfTarget::predict(TimePoint timestamp)
{
  predict(std::chrono::duration<double>(timestamp - t_).count());
}

std::vector<Eigen::Vector4d> EskfTarget::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> result;
  if (!initialized_) {
    return result;
  }
  const int count = armor_num();
  result.reserve(count);
  for (int id = 0; id < count; ++id) {
    const auto pose = VM::armorPose<double>(x_.data(), id, count, name);
    const Eigen::Vector3d ypr = L6Telemetry::rotationToYpr(pose.linear());
    result.emplace_back(
      pose.translation().x(), pose.translation().y(), pose.translation().z(), ypr.x());
  }
  return result;
}

bool EskfTarget::diverged() const
{
  const double r1 = std::exp(x_[VM::idx::LOG_R1]);
  if (!std::isfinite(r1) || r1 < VM::kMinArmorRadius || r1 > VM::kMaxArmorRadius) {
    return true;
  }
  if (name != ArmorName::Outpost) {
    const double r2 = std::exp(x_[VM::idx::LOG_R2]);
    if (!std::isfinite(r2) || r2 < VM::kMinArmorRadius || r2 > VM::kMaxArmorRadius) {
      return true;
    }
  }
  return !x_.allFinite();
}

EskfTarget EskfTarget::snapshot() const
{
  EskfTarget copy;
  copy.config_ = config_;
  copy.armor_config_ = armor_config_;
  copy.x_ = x_;
  copy.t_ = t_;
  copy.name = name;
  copy.armor_type = armor_type;
  copy.jumped = jumped;
  copy.last_id = last_id;
  copy.initialized_ = initialized_;
  copy.converged_ = converged_;
  copy.update_count_ = update_count_;
  // 刻意不复制 filter_：下游拿到的是纯状态副本，外推随便做，不会污染滤波器。
  return copy;
}

}  // namespace L3Estimation
