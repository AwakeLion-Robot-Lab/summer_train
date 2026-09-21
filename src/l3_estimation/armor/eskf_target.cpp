#include "l3_estimation/armor/eskf_target.hpp"

#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace L3Estimation {

namespace VM = VehicleModel;

namespace {

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

}  // namespace

EskfTarget::EskfTarget(
  ArmorName target_name, double x, double vyaw, double radius, double yaw,
  double height_offset, Eigen::Vector3d velocity, EskfTargetConfig config)
{
  config_ = config;
  name = target_name;
  armor_type = armorTypeOf(target_name).value_or(ArmorType::Small);

  x_.setZero();
  x_[VM::idx::CX] = x;
  x_[VM::idx::VYAW] = vyaw;
  x_[VM::idx::ROT_Z] = yaw;
  x_[VM::idx::VCX] = velocity.x();
  x_[VM::idx::VCY] = velocity.y();
  x_[VM::idx::VCZ] = velocity.z();
  const double safe_radius =
    std::clamp(radius, VM::kMinArmorRadius, VM::kMaxArmorRadius);
  x_[VM::idx::LOG_R1] = std::log(safe_radius);
  if (target_name != ArmorName::Outpost) {
    x_[VM::idx::LOG_R2] = std::log(safe_radius);
    x_[VM::idx::HEIGHT] = height_offset;
  }

  t_ = TimePoint{};
  initialized_ = true;
  // 合成目标不带滤波器，没有“更新过多少次”可言，直接当作已收敛，否则下游的
  // converged() 门限会把单测里的目标挡掉。
  converged_ = true;
  jumped = true;
  voter_.reset(t_);
}

void EskfTarget::reset(
  const Armor & armor, const EskfTargetConfig & config, TimePoint timestamp)
{
  config_ = config;
  name = armor.name;
  armor_type = armor.type;

  const double radius = initialRadiusFor(name, config_);

  // 初始协方差按先验给量级。角速度这一维给得最大：单帧根本看不出车在不在转，
  // 等于告诉滤波器这一维基本不知道，放手用观测去改。
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

  // 由这块板的位姿反推整车位姿：T_car = T_armor · (T_armor^car)⁻¹。
  //
  // 右边那一项要知道板编号和半径，这里都用假设值——当成 0 号板（θ=0，板心在
  // 车体系的 (-r, 0, 0)），半径取先验。
  //
  // 两个假设都会错，但都不致命：编号只是标签的循环平移，认错了整车 yaw 差
  // 2πk/N，几何仍然自洽；半径偏差由后续观测修正，p0 给了足够的不确定性。
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
  const Eigen::Vector3d rotation = L6Telemetry::so3Log<double>(vehicle_in_world.linear());
  x_[VM::idx::ROT_X] = rotation.x();
  x_[VM::idx::ROT_Y] = rotation.y();
  x_[VM::idx::ROT_Z] = rotation.z();
  // 速度、角速度、高度差都从零起步。

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

  voter_.reset(timestamp);

  t_ = timestamp;
  initialized_ = true;
  converged_ = false;
  jumped = false;
  last_id = 0;
  update_count_ = 0;
  // 本帧诊断量属于上一次更新；重新初始化的这一帧没有更新，不清掉的话下游会把
  // 旧目标的残差当成新目标的读出去。
  last_nis_ = 0.0;
  last_nis_dof_ = 0;
  last_light_residual_ = LightResidual{};
}

void EskfTarget::predictEkf(TimePoint timestamp, std::optional<TimePoint> hold_from)
{
  if (!filter_) {
    return;
  }
  const double dt = std::chrono::duration<double>(timestamp - t_).count();
  double motion_dt = dt;
  // 预测时长不能超过 hold_from：hold_from 之后的外推不可信，滤波器也不该再往前推进。
  if (hold_from) {
    const double until_hold = std::chrono::duration<double>(*hold_from - t_).count();
    motion_dt = std::min(dt, std::max(until_hold, 0.0));
  }

  filter_->setPredictFunc(
    VM::Motion{.dt = motion_dt, .name = name, .outpost_direction = voter_.sign()});
  // Q 依赖当前姿态（要旋到世界系）和当前半径（log 换算），得在推进前按当时的
  // 状态求值，所以传的是 lambda 而不是一个算好的矩阵。Q 用完整的 dt：原地保持
  // 不等于位置更确定，没观测的这段时间里目标照样可能在动。
  filter_->setUpdateQ([this, dt]() {
    return VM::processNoise(x_, dt, name, config_.noise);
  });

  x_ = filter_->predict();
  t_ = timestamp;
}

int EskfTarget::update(
  const std::vector<MatchedArmor> & matched, const std::vector<MatchedLight> & matched_lights,
  const std::optional<double> & lights_depth_diff, TimePoint timestamp, const ObsContext & ctx)
{
  if (matched.empty() || !filter_) {
    return 0;
  }

  // 观测块的排布固定为 [灯条 4×n][深度差 0 或 1]：灯条在前、连续排放，更新后
  // 按下标就能把扁平残差拆回每根灯条。axes 与灯条块一一对应，记的是检测到的
  // 灯条方向和长度，诊断时拿它把端点残差投到灯条坐标系。
  std::vector<VehicleObs> observations;
  std::vector<LightAxis> axes;
  const std::size_t light_capacity = matched.size() * 2 + matched_lights.size();
  observations.reserve(light_capacity + 1);
  axes.reserve(light_capacity);

  const auto addLight = [&](const cv::Point2f & top, const cv::Point2f & bottom, int id,
                            bool is_left, bool isolated) {
    observations.push_back(lightObs(ctx, top, bottom, id, is_left, isolated));
    axes.push_back(LightAxis{lightDirection(top, bottom), cv::norm(top - bottom)});
  };

  for (const auto & [id, armor] : matched) {
    // 见过 0 号以外的板就粘滞置位，此后整车 yaw 和第二组半径才真正可观测。
    jumped = jumped || (id != 0);
    last_id = id;

    // 把一块完整板拆成左右两根灯条。角点序是左上、右上、右下、左下，所以
    // 左灯条取 [0]、[3]，右灯条取 [1]、[2]。
    addLight(armor.points[0], armor.points[3], id, true, false);
    addLight(armor.points[1], armor.points[2], id, false, false);
  }

  for (const auto & [id, is_left, light] : matched_lights) {
    addLight(light.top, light.bottom, id, is_left, true);
  }

  // 只有一块完整板时，纯重投影观测在斜视方向容易退化，这里补一维 IPPE 给的
  // 左右灯条中心深度差；绝对位姿仍然不写进观测。
  const bool has_depth_diff =
    matched.size() == 1 && lights_depth_diff && std::isfinite(*lights_depth_diff);
  if (has_depth_diff) {
    observations.push_back(makeDepthObs(
      ctx, matched.front().first, *lights_depth_diff,
      config_.armor_lights_depth_diff_sigma));
  }

  x_ = filter_->updateMulti(observations);
  t_ = timestamp;

  // 先验线性化点上的创新量（滤波器在第 0 轮迭代记下），两个诊断量都从它来。
  // NIS 算不出来时保留上一次的值，不要填 0——0 会被读成"这一帧一致性极好"。
  const Eigen::VectorXd & innovation = filter_->lastResidual();
  if (const auto nis = chi2(innovation, filter_->lastInnovCov())) {
    last_nis_ = *nis;
    last_nis_dof_ = static_cast<int>(innovation.size());
  }
  last_light_residual_ = analyzeLight(innovation, axes, has_depth_diff);

  // 拿更新后的整车 yaw 给前哨转向投一票。
  const Eigen::Matrix3d rotation = VM::vehicleRotation<double>(x_.data(), name);
  voter_.update(L6Telemetry::rotationToYpr(rotation).x(), timestamp);

  update_count_ += static_cast<int>(observations.size());
  if (update_count_ > 20 && !diverged()) {
    converged_ = true;
  }
  return static_cast<int>(observations.size());
}

ObsContext EskfTarget::obsContext(
  const L1Sensor::CameraCalibration & calibration,
  const Eigen::Isometry3d & camera_in_world) const
{
  ObsContext ctx;
  ctx.name = name;
  ctx.armor_num = armor_num();
  ctx.armor = config_.armor;
  ctx.camera_in_world = camera_in_world;
  ctx.camera_matrix = calibration.camera_matrix;
  ctx.distortion_coefficients = calibration.distortion_coefficients;
  return ctx;
}

Eigen::VectorXd EskfTarget::stateAt(TimePoint timestamp) const
{
  EskfTarget predicted = snapshot();
  predicted.predict(timestamp);
  return predicted.x_;
}

std::pair<double, double> EskfTarget::lightSigma(double length, bool isolated) const
{
  const double scale = isolated ? config_.isolated_light_sigma_scale : 1.0;
  return {
    std::max(config_.sigma_min_px, config_.sigma_along_by_length * length) * scale,
    std::max(config_.sigma_min_px, config_.sigma_perp_by_length * length) * scale};
}

VehicleObs EskfTarget::lightObs(
  const ObsContext & ctx, const cv::Point2f & top, const cv::Point2f & bottom, int id,
  bool is_left, bool isolated) const
{
  const auto [sigma_along, sigma_perp] = lightSigma(cv::norm(top - bottom), isolated);
  return makeLightObs(ctx, top, bottom, id, is_left, sigma_along, sigma_perp);
}

std::optional<double> EskfTarget::mahalanobis(const VehicleObs & obs) const
{
  if (!filter_ || !obs) {
    return std::nullopt;
  }
  Eigen::VectorXd innovation;
  Eigen::MatrixXd covariance;
  filter_->innovation(*obs, innovation, covariance);
  return chi2(innovation, covariance);
}

Eigen::VectorXd EskfTarget::ekf_x() const
{
  Eigen::VectorXd out = x_;
  // 半径对外吐线性值，内部的 log 表示不泄漏给 L4。
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
  copy.x_ = x_;
  copy.t_ = t_;
  copy.name = name;
  copy.armor_type = armor_type;
  copy.jumped = jumped;
  copy.last_id = last_id;
  copy.initialized_ = initialized_;
  copy.converged_ = converged_;
  copy.update_count_ = update_count_;
  copy.last_nis_ = last_nis_;
  copy.last_nis_dof_ = last_nis_dof_;
  // 端点残差和 NIS 一样是本帧诊断量，必须跟着副本走：track() 对外返回的就是
  // snapshot，漏掉它 track_diag 的创新列会恒为空。
  copy.last_light_residual_ = last_light_residual_;
  copy.voter_ = voter_;
  // 刻意不复制 filter_：下游拿到的是纯状态副本，外推随便做，不会污染滤波器。
  return copy;
}

}  // namespace L3Estimation
