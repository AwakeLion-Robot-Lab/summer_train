#include "l3_estimation/armor/eskf_target.hpp"

#include "l6_telemetry/math.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace L3Estimation {

namespace VM = VehicleModel;

namespace {

// 装甲板四角在图像上的顺序：左上、右上、右下、左下。预测侧也按这个顺序由
// 左右灯条的端点拼出来，两边不一致的话四边形代价算的是两个不同形状。
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

// 算一块板正对相机的程度：板的 x 轴指向车心，朝外的法向是 -axis_x，它与
// 「板 → 相机」方向的点积越大，板越正对。
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

EskfTarget::EskfTarget(
  ArmorName target_name, double x, double vyaw, double radius, double yaw,
  double height_offset, Eigen::Vector3d velocity, EskfTargetConfig config)
{
  config_ = config;
  armor_config_ = config.armor;
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

// 相机光学系在世界系下的位姿：由 camera -> barrel 的静态外参左乘当帧的
// barrel -> world 旋转得到，世界系原点取枪管原点。
Eigen::Isometry3d EskfTarget::cameraInWorld(
  const L1Sensor::CameraCalibration & calibration, const Eigen::Quaterniond & q_world_barrel)
{
  Eigen::Isometry3d barrel_in_world = Eigen::Isometry3d::Identity();
  barrel_in_world.linear() = q_world_barrel.toRotationMatrix();
  if (!calibration.T_barrel_camera) {
    return barrel_in_world;
  }
  return barrel_in_world * (*calibration.T_barrel_camera);
}

void EskfTarget::predictEkf(TimePoint timestamp, std::optional<TimePoint> hold_from)
{
  if (!filter_) {
    return;
  }
  const double dt = std::chrono::duration<double>(timestamp - t_).count();
  double motion_dt = dt;
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

LightContext EskfTarget::makeContext(
  int id, bool is_left, const L1Sensor::CameraCalibration & calibration,
  const Eigen::Isometry3d & camera_in_world) const
{
  LightContext ctx;
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
  const LightMeasure measure{makeContext(id, is_left, calibration, camera_in_world)};
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

  // 先把状态外推到本帧曝光时刻，再拿预测位置做关联。
  EskfTarget predicted = snapshot();
  predicted.predict(timestamp);
  const Eigen::VectorXd state = predicted.x_;

  // 可见性筛选：按正对相机的程度排序，只留最正对的三块。四板车最多同时看到
  // 两块半，取三留了余量。这只是几何近似，判的是板朝不朝着你，不判它有没有
  // 被车身自己挡住。
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

      // 拼出预测四边形，顺序与检测角点一致：左上、右上、右下、左下。
      const std::array<cv::Point2f, kCorners> predicted_corners{
        left.first, right.first, right.second, left.second};

      // 代价是四边形与四边形的差异，三项各描述一个自由度：
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

std::vector<EskfTarget::MatchedLight> EskfTarget::matchLight(
  const std::vector<L2Perception::Light>& lights,
  const std::vector<std::pair<int, Armor>>& matched_armors,
  TimePoint timestamp, const L1Sensor::CameraCalibration& calibration,
  const Eigen::Isometry3d& camera_in_world) const
{
  std::vector<MatchedLight> result;
  const bool is_base =
    name == ArmorName::BaseSmall || name == ArmorName::BaseLarge;
  // 本帧至少关联上一块完整板才做：没有完整板做锚，侧边灯条的编号和左右归属
  // 几乎是猜的。基地的板不绕转，整车预测也约束不了它的灯条位置。
  if (!config_.enable_lights_measure || is_base || matched_armors.empty() ||
      lights.empty() || !initialized_ || !filter_ ||
      (config_.light_match_require_jumped && !jumped)) {
    return result;
  }

  EskfTarget predicted = snapshot();
  predicted.predict(timestamp);
  const Eigen::VectorXd state = predicted.x_;
  const int count = armor_num();

  std::vector<double> facing(count, 0.0);
  for (int id = 0; id < count; ++id) {
    const Eigen::Isometry3d pose_in_world =
      VM::armorPose<double>(state.data(), id, count, name);
    facing[id] = facingScore(camera_in_world.inverse() * pose_in_world);
  }
  const int closest_id = static_cast<int>(
    std::max_element(facing.begin(), facing.end()) - facing.begin());

  using PredictedLight =
    std::tuple<int, bool, std::pair<cv::Point2f, cv::Point2f>>;
  std::vector<PredictedLight> visible_lights;
  visible_lights.reserve(4);
  const auto matchedPlate = [&matched_armors](int id) {
    return std::any_of(
      matched_armors.begin(), matched_armors.end(),
      [id](const std::pair<int, Armor>& matched) { return matched.first == id; });
  };
  const auto addVisible = [&](int id, bool is_left) {
    // 已配成完整板的板，两根灯条本帧都由 update() 从板的角点拆出来，不再需要
    // 侧边灯条补位；背对相机的板看不到灯条，槽位留着只会招来错配。
    if (matchedPlate(id) || !(facing[id] > 0.0)) {
      return;
    }
    visible_lights.emplace_back(
      id, is_left,
      predicted.predictLight(id, is_left, state, calibration, camera_in_world));
  };

  // 候选限定在最正对那块板的左右灯条，加上相邻两块板靠近它的各一根。
  addVisible((closest_id + count - 1) % count, false);
  addVisible((closest_id + 1) % count, true);
  addVisible(closest_id, false);
  addVisible(closest_id, true);

  if (visible_lights.empty()) {
    return result;
  }

  constexpr double kMaxCost = 1e9;
  const int observation_count = static_cast<int>(lights.size());
  std::vector<std::vector<double>> cost(
    observation_count,
    std::vector<double>(visible_lights.size(), kMaxCost + 1.0));

  const auto lightCost = [&](const L2Perception::Light& light,
                             const PredictedLight& candidate) -> double {
    const auto& [id, is_left, endpoints] = candidate;
    const auto& [top, bottom] = endpoints;
    const double predicted_length = cv::norm(top - bottom);
    if (!(predicted_length > 1e-6)) {
      return kMaxCost + 1.0;
    }

    const double length_error = std::abs(light.length - predicted_length);
    if (length_error > predicted_length * config_.light_match_length_ratio_gate) {
      return kMaxCost + 1.0;
    }

    // 角度取 atan2(Δx, Δy)，预测和检测同一约定，差值再折回 (-π, π]。
    const double predicted_angle = std::atan2(top.x - bottom.x, top.y - bottom.y);
    const double light_angle =
      std::atan2(light.top.x - light.bottom.x, light.top.y - light.bottom.y);
    const double angle_error =
      std::abs(VM::normalizeAngle(light_angle - predicted_angle));
    if (angle_error > config_.light_match_angle_gate) {
      return kMaxCost + 1.0;
    }

    // 马氏距离用的 S 与这根灯条若被采纳时真正进更新的那份一致。
    const auto obs = lightObs(
      light.top, light.bottom, id, is_left, true, calibration, camera_in_world);
    Eigen::VectorXd innovation;
    Eigen::MatrixXd innovation_covariance;
    filter_->innovation(*obs, innovation, innovation_covariance);
    const Eigen::LLT<Eigen::MatrixXd> llt(innovation_covariance);
    if (llt.info() != Eigen::Success) {
      return kMaxCost + 1.0;
    }
    const double distance = innovation.dot(llt.solve(innovation));
    if (!(distance <= config_.light_match_chi2_gate)) {
      return kMaxCost + 1.0;
    }
    return distance;
  };

  for (int observation = 0; observation < observation_count; ++observation) {
    for (std::size_t candidate = 0; candidate < visible_lights.size(); ++candidate) {
      cost[observation][candidate] =
        lightCost(lights[observation], visible_lights[candidate]);
    }
  }

  for (const auto& [observation, candidate] : greedyMatch(
         cost, observation_count, static_cast<int>(visible_lights.size()), kMaxCost)) {
    const auto& [id, is_left, unused] = visible_lights[candidate];
    result.emplace_back(id, is_left, lights[observation]);
  }
  return result;
}

std::shared_ptr<EskfTarget::Filter::ObsBase> EskfTarget::lightObs(
  const cv::Point2f & top, const cv::Point2f & bottom, int id, bool is_left, bool isolated,
  const L1Sensor::CameraCalibration & calibration,
  const Eigen::Isometry3d & camera_in_world) const
{
  const LightMeasure measure{makeContext(id, is_left, calibration, camera_in_world)};
  const double length = cv::norm(top - bottom);
  const double scale = isolated ? config_.isolated_light_sigma_scale : 1.0;
  const double sigma_along =
    std::max(config_.sigma_min_px, config_.sigma_along_by_length * length) * scale;
  const double sigma_perp =
    std::max(config_.sigma_min_px, config_.sigma_perp_by_length * length) * scale;
  const LightCov r_cov = lightCov(top, bottom, sigma_along, sigma_perp);

  return Filter::makeObs<kLightMeasureSize>(
    toLight(top, bottom), measure, [r_cov](const LightVector &) { return r_cov; },
    [](const LightVector & z_pred, const LightVector & z_obs) {
      return LightMeasure::residual<double>(z_pred, z_obs);
    });
}

int EskfTarget::update(
  const std::vector<std::pair<int, Armor>> & matched, TimePoint timestamp,
  const L1Sensor::CameraCalibration & calibration, const Eigen::Isometry3d & camera_in_world)
{
  return update(matched, {}, std::nullopt, timestamp, calibration, camera_in_world);
}

int EskfTarget::update(
  const std::vector<std::pair<int, Armor>>& matched,
  const std::vector<MatchedLight>& matched_lights,
  const std::optional<double>& lights_depth_diff, TimePoint timestamp,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Isometry3d& camera_in_world)
{
  if (matched.empty() || !filter_) {
    return 0;
  }

  std::vector<std::shared_ptr<Filter::ObsBase>> observations;
  // 观测块的排布固定为 [灯条 4×n][深度差 0 或 1]：灯条在前、连续排放，更新后
  // 按下标就能把扁平残差拆回每根灯条。directions 与灯条块一一对应，记的是
  // 检测到的灯条方向，诊断时拿它把端点残差投到灯条坐标系。
  std::vector<Eigen::Vector2d> directions;
  directions.reserve(matched.size() * 2 + matched_lights.size());

  const auto addLight = [&](const cv::Point2f & top, const cv::Point2f & bottom, int id,
                            bool is_left, bool isolated) {
    observations.push_back(
      lightObs(top, bottom, id, is_left, isolated, calibration, camera_in_world));
    directions.push_back(lightDirection(top, bottom));
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

  for (const auto& [id, is_left, light] : matched_lights) {
    addLight(light.top, light.bottom, id, is_left, true);
  }
  const int light_count = static_cast<int>(directions.size());

  // 只有一块完整板时，纯重投影观测在斜视方向容易退化，这里补一维 IPPE 给的
  // 左右灯条中心深度差；绝对位姿仍然不写进观测。
  bool has_depth_diff = false;
  if (matched.size() == 1 && lights_depth_diff &&
      std::isfinite(*lights_depth_diff)) {
    const int id = matched.front().first;
    const DepthDiffMeasure measure{
      makeContext(id, true, calibration, camera_in_world)};
    DepthDiffVector z;
    z[0] = *lights_depth_diff;

    Eigen::Matrix<double, kDepthDiffMeasureSize, kDepthDiffMeasureSize> r_cov;
    r_cov.setZero();
    const double sigma = config_.armor_lights_depth_diff_sigma;
    r_cov(0, 0) = sigma * sigma / 2.0;

    observations.push_back(Filter::makeObs<kDepthDiffMeasureSize>(
      z, measure, [r_cov](const DepthDiffVector&) { return r_cov; },
      [](const DepthDiffVector& z_pred, const DepthDiffVector& z_obs) {
        return DepthDiffMeasure::residual<double>(z_pred, z_obs);
      }));
    has_depth_diff = true;
  }

  x_ = filter_->updateMulti(observations);
  t_ = timestamp;

  // NIS = rᵀ S⁻¹ r，r 和 S 都取先验线性化点上的（滤波器在第 0 轮迭代记下）。
  const Eigen::VectorXd & innovation = filter_->lastResidual();
  const Eigen::MatrixXd & innovation_covariance = filter_->lastInnovCov();
  if (innovation.size() > 0 && innovation_covariance.rows() == innovation.size()) {
    const Eigen::LLT<Eigen::MatrixXd> llt(innovation_covariance);
    if (llt.info() == Eigen::Success) {
      last_nis_ = innovation.dot(llt.solve(innovation));
      last_nis_dof_ = static_cast<int>(innovation.size());
    }
  }

  // 把每根灯条两个端点的残差投到该灯条的方向 e 和法向 n 上，分方向累加。
  last_light_residual_ = LightResidual{};
  if (innovation.size() >= light_count * kLightMeasureSize) {
    double along_sq = 0.0;
    double perp_sq = 0.0;
    for (int i = 0; i < light_count; ++i) {
      const int base = i * kLightMeasureSize;
      const Eigen::Vector2d & e = directions[i];
      const Eigen::Vector2d n(-e.y(), e.x());
      for (const int point : {endpoint::TOP_U, endpoint::BOTTOM_U}) {
        const Eigen::Vector2d r = innovation.segment<2>(base + point);
        along_sq += r.dot(e) * r.dot(e);
        perp_sq += r.dot(n) * r.dot(n);
      }
    }
    if (light_count > 0) {
      const double samples = 2.0 * light_count;
      last_light_residual_.along_rms_px = std::sqrt(along_sq / samples);
      last_light_residual_.perp_rms_px = std::sqrt(perp_sq / samples);
      last_light_residual_.light_count = light_count;
    }
    const int depth_index = light_count * kLightMeasureSize;
    if (has_depth_diff && depth_index < innovation.size()) {
      last_light_residual_.depth_diff_m = innovation[depth_index];
    }
  }

  // 拿更新后的整车 yaw 给前哨转向投一票。
  const Eigen::Matrix3d rotation = VM::vehicleRotation<double>(x_.data(), name);
  voter_.update(L6Telemetry::rotationToYpr(rotation).x(), timestamp);

  update_count_ += static_cast<int>(observations.size());
  if (update_count_ > 20 && !diverged()) {
    converged_ = true;
  }
  return static_cast<int>(observations.size());
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
