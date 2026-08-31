#include "l3_estimation/armor/eskf_tracker.hpp"

#include "l3_estimation/armor/tracker.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace L3Estimation {

namespace {

bool validTrackerConfig(const EskfTrackerConfig & config) noexcept
{
  return config.tracking_thres > 0 && config.lost_time_thres > 0.0 &&
         config.lost_time_thres_outpost >= config.lost_time_thres;
}

}  // namespace

EskfTracker::EskfTracker(
  const L1Sensor::CameraCalibration & calibration, ArmorConfig armor_config,
  EskfTrackerConfig tracker_config, EskfTargetConfig target_config)
: calibration_(calibration),
  armor_config_(armor_config),
  tracker_config_(tracker_config),
  target_config_(std::move(target_config)),
  pnp_solver_(calibration, armor_config),
  image_center_{
    static_cast<float>(calibration.image_size.width) / 2.0F,
    static_cast<float>(calibration.image_size.height) / 2.0F},
  ready_(pnp_solver_.ready() && validTrackerConfig(tracker_config))
{
  // 板尺寸只在一处配置，避免 PnP 物点与 UVL 灯条端点用了两套几何。
  target_config_.armor = armor_config_;
}

void EskfTracker::reset() noexcept
{
  for (auto & slot : buffer_) {
    slot.lifecycle.reset();
    slot.target = EskfTarget{};
    slot.uvl_update_lights.clear();
  }
  current_ = 0;
  previous_ = 1;
  observations_.clear();
}

bool EskfTracker::semanticObservationUsable(const Armor& armor) const noexcept
{
  return armor.name != ArmorName::Unknown &&
         armor.color != L2Perception::ArmorColor::Unknown &&
         std::all_of(
           armor.points.begin(), armor.points.end(),
           [](const cv::Point2f& point) {
             return std::isfinite(point.x) && std::isfinite(point.y);
           });
}

bool EskfTracker::pnpObservationUsable(const Armor& armor) const noexcept
{
  return armor.name != ArmorName::Unknown && armor.xyz_in_camera.allFinite() &&
         armor.xyz_in_camera.z() > 0.0 && armor.xyz_in_world.allFinite() &&
         armor.ypr_in_camera.allFinite() &&
         std::isfinite(armor.reprojection_error);
}

std::vector<Armor> EskfTracker::initializationObservations()
{
  std::vector<Armor> result;
  result.reserve(observations_.size());
  for (const Armor& observation : observations_) {
    if (!semanticObservationUsable(observation)) {
      continue;
    }
    Armor pnp_observation = observation;
    pnp_solver_.single_pnp(pnp_observation);
    if (pnpObservationUsable(pnp_observation)) {
      result.push_back(std::move(pnp_observation));
    }
  }
  // 初始化时优先取离图像中心最近的，那通常是操作手正对着的目标。
  std::sort(result.begin(), result.end(), [this](const Armor& lhs, const Armor& rhs) {
    return L6Telemetry::squaredDistance(lhs.center, image_center_) <
           L6Telemetry::squaredDistance(rhs.center, image_center_);
  });
  return result;
}

double EskfTracker::lostTimeThreshold(const EskfTarget & target) const noexcept
{
  return target.name == ArmorName::Outpost ? tracker_config_.lost_time_thres_outpost
                                           : tracker_config_.lost_time_thres;
}

bool EskfTracker::initializeTarget(
  Slot& slot, const std::vector<Armor>& candidates, TimePoint timestamp,
  const Eigen::Isometry3d & camera_in_world)
{
  slot.uvl_update_lights.clear();
  if (candidates.empty()) {
    return false;
  }
  // 已排序，取最靠近图像中心的那块。
  const Armor& selected = candidates.front();
  slot.target.reset(selected, target_config_, timestamp, calibration_, camera_in_world);
  slot.lifecycle.state = TrackState::Detecting;
  slot.lifecycle.detect_count = 0;
  slot.last_update = timestamp;
  return true;
}

bool EskfTracker::updateTarget(
  Slot& slot, const std::vector<Armor>& candidates,
  const std::vector<L2Perception::Light>& lights, TimePoint timestamp,
  const Eigen::Isometry3d & camera_in_world)
{
  slot.uvl_update_lights.clear();

  // 只有同类别的板才可能属于同一辆车。
  std::vector<Armor> same_name;
  same_name.reserve(candidates.size());
  for (const auto & armor : candidates) {
    if (armor.name == slot.target.name && semanticObservationUsable(armor)) {
      same_name.push_back(armor);
    }
  }

  slot.target.predictEkf(timestamp);
  const auto matched =
    slot.target.matchArmor(same_name, timestamp, calibration_, camera_in_world);
  const auto matched_lights = slot.target.matchLight(
    lights, matched, timestamp, calibration_, camera_in_world);

  std::optional<double> depth_difference;
  if (matched.size() == 1) {
    depth_difference = pnp_solver_.armor_lights_depth_difference(
      matched.front().second);
  }

  last_match_count_ = static_cast<int>(matched.size());
  last_matched_ids_.clear();
  for (const auto& [id, armor] : matched) {
    if (!last_matched_ids_.empty()) {
      last_matched_ids_ += '|';
    }
    last_matched_ids_ += std::to_string(id);
  }

  const int updated = slot.target.update(
    matched, matched_lights, depth_difference, timestamp, calibration_,
    camera_in_world);

  if (updated > 0) {
    // 只有 update() 成功之后才发布，用于显示的清单因此严格等于本帧真正进入
    // updateMulti() 的 UVL 观测，而不是 matchLight() 的候选或关联中间结果。
    slot.uvl_update_lights.reserve(
      matched.size() * 2 + matched_lights.size());
    for (const auto& [id, armor] : matched) {
      slot.uvl_update_lights.push_back(
        {armor.points[0], armor.points[3], id, true, false});
      slot.uvl_update_lights.push_back(
        {armor.points[1], armor.points[2], id, false, false});
    }
    for (const auto& [id, is_left, light] : matched_lights) {
      slot.uvl_update_lights.push_back(
        {light.top, light.bottom, id, is_left, true});
    }
    slot.last_update = timestamp;
  }
  return updated > 0;
}

std::optional<EskfTarget> EskfTracker::track(
  const std::vector<L2Perception::Armor> & detections,
  const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp)
{
  return track(detections, {}, q_world_barrel, timestamp);
}

std::optional<EskfTarget> EskfTracker::track(
  const std::vector<L2Perception::Armor>& detections,
  const std::vector<L2Perception::Light>& lights,
  const std::optional<Eigen::Quaterniond>& q_world_barrel,
  TimePoint timestamp)
{
  observations_.clear();
  last_match_count_ = 0;
  last_matched_ids_.clear();
  // 每帧先清空两个槽。后面只有实际完成滤波更新的槽会重新填充，早退、初始化
  // 或 TempLost 纯预测时不会泄漏上一帧的显示结果。
  for (auto& slot : buffer_) {
    slot.uvl_update_lights.clear();
  }
  if (!ready_ || !q_world_barrel) {
    return std::nullopt;
  }

  // L2 -> L3：正常更新只搬运类别和四角点，不以 PnP 成功作为入口门限。
  // PnP 仅在 Lost 初始化和单完整板深度差约束时调用，与 Awakening 一致。
  pnp_solver_.set_R_world_barrel(q_world_barrel);
  observations_.reserve(detections.size());
  for (const auto & detection : detections) {
    Armor observation = toArmorObservation(detection, timestamp);
    observation.name = L2Perception::armorClassFromId(observation.class_id);
    if (const auto type = armorTypeOf(observation.name)) {
      observation.type = *type;
    }
    observations_.push_back(std::move(observation));
  }

  const Eigen::Isometry3d camera_in_world =
    EskfTarget::cameraInWorld(calibration_, *q_world_barrel);

  // 初始化候选按需计算并在本帧缓存：正常 Tracking 更新完全不跑 PnP；若当前槽
  // 恰在这一帧转入 TempLost，下面立刻处理备用 Lost 槽时仍能与 Awakening 一样
  // 当帧完成初始化，而不是平白晚一帧。
  std::optional<std::vector<Armor>> initialization_candidates;
  const auto getInitializationCandidates = [&]() -> const std::vector<Armor>& {
    if (!initialization_candidates) {
      initialization_candidates = initializationObservations();
    }
    return *initialization_candidates;
  };

  const auto process = [&](std::size_t index) {
    Slot & slot = buffer_[index];
    const bool found = (slot.lifecycle.state == TrackState::Lost)
                         ? initializeTarget(
                             slot, getInitializationCandidates(), timestamp,
                             camera_in_world)
                         : updateTarget(
                             slot, observations_, lights, timestamp,
                             camera_in_world);

    updateFsm(
      found, slot.lifecycle, tracker_config_.tracking_thres,
      elapsedSeconds(slot.last_update, timestamp), lostTimeThreshold(slot.target));

    // 发散的目标直接丢弃，不要让它把下游一起拖歪。
    if (slot.lifecycle.isTracking() && slot.target.diverged()) {
      slot.lifecycle.reset();
      slot.uvl_update_lights.clear();
      L6Telemetry::logWarn("EskfTracker: 目标发散，已复位");
    }
    return found;
  };

  // 双缓冲：当前目标进 TempLost 时，另一个槽位同时尝试抓新目标；新目标一旦
  // 转成 Tracking 就顶上，不必等当前目标超时。
  process(current_);

  Slot & current = buffer_[current_];
  Slot & previous = buffer_[previous_];

  if (current.lifecycle.state == TrackState::TempLost) {
    process(previous_);
    if (previous.lifecycle.state == TrackState::Tracking) {
      std::swap(current, previous);
      previous.lifecycle.reset();
    }
  } else if (current.lifecycle.state == TrackState::Tracking) {
    previous.lifecycle.reset();
  }

  const Slot & active = buffer_[current_];
  if (active.lifecycle.state == TrackState::Lost || !active.target.initialized()) {
    return std::nullopt;
  }
  // 下游拿到的是不含滤波器的副本，随便外推不会污染滤波器状态。
  return active.target.snapshot();
}

std::optional<cv::Rect> EskfTracker::predictedLightBounds(
  const std::optional<Eigen::Quaterniond>& q_world_barrel, TimePoint timestamp,
  const cv::Size& image_size, bool require_light_measurements) const
{
  if (!q_world_barrel || image_size.width <= 0 || image_size.height <= 0) {
    return std::nullopt;
  }
  const Slot& active = buffer_[current_];
  if (!active.lifecycle.isTracking() || !active.target.initialized() ||
      elapsedSeconds(active.last_update, timestamp) >=
        lostTimeThreshold(active.target)) {
    return std::nullopt;
  }
  // 传统灯条检测那条路额外要求：开关打开，且目标不是基地（基地板不绕转，
  // 整车预测对它的灯条位置没有约束力）。
  if (require_light_measurements &&
      (!active.target.lightMeasurementsEnabled() ||
       active.target.name == ArmorName::BaseSmall ||
       active.target.name == ArmorName::BaseLarge)) {
    return std::nullopt;
  }

  EskfTarget predicted = active.target.snapshot();
  predicted.predict(timestamp);
  const Eigen::VectorXd state = predicted.rawState();
  const Eigen::Isometry3d camera_in_world =
    EskfTarget::cameraInWorld(calibration_, *q_world_barrel);

  std::vector<cv::Point2f> points;
  points.reserve(static_cast<std::size_t>(predicted.armor_num()) * 4);
  for (int id = 0; id < predicted.armor_num(); ++id) {
    for (const bool is_left : {true, false}) {
      const auto light = predicted.predictLight(
        id, is_left, state, calibration_, camera_in_world);
      for (const cv::Point2f& point : {light.first, light.second}) {
        if (std::isfinite(point.x) && std::isfinite(point.y)) {
          points.push_back(point);
        }
      }
    }
  }
  if (points.empty()) {
    return std::nullopt;
  }

  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  const cv::Rect bounds = cv::boundingRect(points);
  if ((bounds & image_rect).empty()) {
    return std::nullopt;
  }
  return bounds;
}

namespace {

// 以中心不动的方式按比例缩放矩形，再裁到图像内。
cv::Rect expandAndClip(const cv::Rect& rect, double ratio, const cv::Rect& image_rect)
{
  const double center_x = rect.x + rect.width * 0.5;
  const double center_y = rect.y + rect.height * 0.5;
  const int width = std::max(1, static_cast<int>(std::round(rect.width * ratio)));
  const int height = std::max(1, static_cast<int>(std::round(rect.height * ratio)));
  cv::Rect expanded(
    static_cast<int>(std::round(center_x - width * 0.5)),
    static_cast<int>(std::round(center_y - height * 0.5)), width, height);
  return expanded & image_rect;
}

}  // namespace

std::optional<cv::Rect> EskfTracker::lightDetectionRoi(
  const std::optional<Eigen::Quaterniond>& q_world_barrel,
  TimePoint timestamp, const cv::Size& image_size) const
{
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  const auto bounds = predictedLightBounds(q_world_barrel, timestamp, image_size, true);
  if (!bounds) {
    return std::nullopt;
  }

  // 送给传统检测的 ROI 越紧越好：搜索区域越大，环境灯光越容易混进候选集，
  // CPU 开销和误匹配概率也一起上去。
  constexpr double kExpandRatio = 1.6;
  const cv::Rect expanded = expandAndClip(*bounds, kExpandRatio, image_rect);
  return expanded.empty() ? std::optional<cv::Rect>{image_rect}
                          : std::optional<cv::Rect>{expanded};
}

cv::Rect EskfTracker::netFocusRoi(
  const std::optional<Eigen::Quaterniond>& q_world_barrel, TimePoint timestamp,
  const cv::Size& image_size, double target_wh_ratio) const
{
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  // 不可聚焦时返回整图，而不是空：调用方拿到的永远是能直接用的矩形。
  const auto bounds = predictedLightBounds(q_world_barrel, timestamp, image_size, false);
  if (!bounds) {
    return image_rect;
  }

  const Slot& active = buffer_[current_];
  const bool is_base = active.target.name == ArmorName::BaseSmall ||
                       active.target.name == ArmorName::BaseLarge;
  // 基地目标本身尺寸大、整车模型退化，ROI 要放得更宽。
  constexpr double kExpandRatio = 1.4;
  constexpr double kExpandRatioBase = 3.0;
  cv::Rect rect =
    expandAndClip(*bounds, is_base ? kExpandRatioBase : kExpandRatio, image_rect);
  if (rect.empty()) {
    return image_rect;
  }

  // ① 按网络输入宽高比修正。工业相机图像的长宽比通常与网络输入不一致，直接
  //    letterbox 会整体缩小；先把 ROI 修成同一比例，padding 就少了。
  const double ratio =
    (std::isfinite(target_wh_ratio) && target_wh_ratio > 0.0) ? target_wh_ratio : 1.0;
  double target_width = std::max(rect.width, 1);
  double target_height = std::max(rect.height, 1);
  if (target_width / target_height < ratio) {
    target_width = target_height * ratio;
  } else {
    target_height = target_width / ratio;
  }
  const double center_x = rect.x + rect.width * 0.5;
  const double center_y = rect.y + rect.height * 0.5;
  cv::Rect ratio_rect(
    static_cast<int>(std::round(center_x - target_width * 0.5)),
    static_cast<int>(std::round(center_y - target_height * 0.5)),
    static_cast<int>(std::round(target_width)),
    static_cast<int>(std::round(target_height)));
  ratio_rect &= image_rect;
  if (ratio_rect.empty()) {
    return image_rect;
  }

  // ② 随丢失时长线性膨胀，超时退化为整图。目标越久没更新，预测越不可信，
  //    搜索范围就该越大——这条让 ROI 机制在跟丢时自动放手，而不是把网络
  //    永远锁在一个错误的小窗口里。
  const double lost_time = elapsedSeconds(active.last_update, timestamp);
  const double lost_thres = lostTimeThreshold(active.target);
  const int base_side = std::max(ratio_rect.width, ratio_rect.height);
  const int max_side = std::max(image_size.width, image_size.height);
  int side = max_side;
  if (lost_thres > 0.0 && lost_time < lost_thres) {
    const double fraction = std::clamp(lost_time / lost_thres, 0.0, 1.0);
    side = static_cast<int>(
      std::round(base_side + (max_side - base_side) * fraction));
  }
  side = std::clamp(side, 1, max_side);

  // ③ 扩成方形，适配常见的方形网络输入。
  const int square_center_x = ratio_rect.x + ratio_rect.width / 2;
  const int square_center_y = ratio_rect.y + ratio_rect.height / 2;
  cv::Rect square(
    square_center_x - side / 2, square_center_y - side / 2, side, side);
  square &= image_rect;
  return square.empty() ? image_rect : square;
}

std::vector<Eigen::Vector4d> EskfTracker::targetArmorPoses() const
{
  const Slot & active = buffer_[current_];
  if (active.lifecycle.state == TrackState::Lost || !active.target.initialized()) {
    return {};
  }
  return active.target.armor_xyza_list();
}

}  // namespace L3Estimation
