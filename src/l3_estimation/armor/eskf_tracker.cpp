#include "l3_estimation/armor/eskf_tracker.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

namespace L3Estimation {

namespace {

bool validTrackerConfig(const EskfTrackerConfig & config) noexcept
{
  return config.tracking_thres > 0 && config.lost_time_thres > 0.0 &&
         config.lost_time_thres_outpost >= config.lost_time_thres &&
         config.max_frame_gap > 0.0 && config.temp_lost_predict_time > 0.0;
}

// L2 -> L3：正常更新只搬类别和四角点，不拿 PnP 成功当入口门限。PnP 只在
// Lost 初始化和单块完整板求深度差时才跑。
std::vector<Armor> toObservations(const std::vector<L2Perception::Armor> & detections)
{
  std::vector<Armor> observations;
  observations.reserve(detections.size());
  for (const auto & detection : detections) {
    Armor observation = toObservation(detection);
    observation.name = L2Perception::armorClassFromId(observation.class_id);
    if (const auto type = armorTypeOf(observation.name)) {
      observation.type = *type;
    }
    observations.push_back(std::move(observation));
  }
  return observations;
}

}  // namespace

Armor toObservation(const L2Perception::Armor & detection)
{
  // 只搬检测字段，三维位姿留给当帧的 PnpSolver 填。
  Armor observation;
  observation.class_id = detection.class_id;
  observation.color = detection.color;
  observation.points = detection.corners;
  observation.center = detection.center;
  observation.confidence = static_cast<double>(detection.confidence);
  observation.area = L6Telemetry::polygonArea(detection.corners);
  return observation;
}

EskfTracker::EskfTracker(
  const L1Sensor::CameraCalibration & calibration, ArmorConfig armor_config,
  EskfTrackerConfig tracker_config, EskfTargetConfig target_config)
: calibration_(calibration),
  tracker_config_(tracker_config),
  target_config_(std::move(target_config)),
  pnp_solver_(calibration, armor_config),
  image_center_{
    static_cast<float>(calibration.image_size.width) / 2.0F,
    static_cast<float>(calibration.image_size.height) / 2.0F},
  ready_(pnp_solver_.ready() && validTrackerConfig(tracker_config))
{
  // 板尺寸只认这一份，免得 PnP 物点和灯条端点观测用上两套几何。
  target_config_.armor = armor_config;
}

void EskfTracker::reset() noexcept
{
  active_ = Slot{};
  backup_ = Slot{};
  last_frame_.reset();
}

// --- 一帧的主流程 ---------------------------------------------------------

std::optional<EskfTarget> EskfTracker::track(
  const std::vector<L2Perception::Armor> & detections,
  const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp)
{
  return track(detections, {}, q_world_barrel, timestamp);
}

std::optional<EskfTarget> EskfTracker::track(
  const std::vector<L2Perception::Armor> & detections,
  const std::vector<L2Perception::Light> & lights,
  const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp)
{
  clearFrame();
  if (!ready_ || !q_world_barrel) {
    L6Telemetry::logWarn("EskfTracker: 未就绪或缺少枪管姿态，无法跟踪");
    return std::nullopt;
  }

  dropOnGap(timestamp);
  last_frame_ = timestamp;
  // PnP 只在初始化候选和单板深度差里跑，两处都要这一帧曝光时刻的枪管姿态。
  pnp_solver_.set_R_world_barrel(q_world_barrel);

  Frame frame{
    .timestamp = timestamp,
    .camera_in_world = cameraInWorld(calibration_, *q_world_barrel),
    .armors = toObservations(detections),
    .lights = &lights};

  const bool was_active = active_.lifecycle.state != TrackState::Lost;
  advance(active_, frame, std::nullopt);
  runBackup(frame);

  // 当前槽本帧刚退回 Lost（Detecting 丢帧、TempLost 超时或发散）：就地拿本帧
  // 的检测重建，不空等一帧。备用槽的任务随之结束，免得留着一个过期的
  // Detecting 目标。
  if (was_active && active_.lifecycle.state == TrackState::Lost) {
    ++drop_count_;
    backup_.lifecycle.reset();
    advance(active_, frame, std::nullopt);
  }

  if (active_.lifecycle.state == TrackState::Lost || !active_.target.initialized()) {
    return std::nullopt;
  }
  // 返回不含滤波器的副本，下游随便外推都不会污染滤波器状态。
  return active_.target.snapshot();
}

void EskfTracker::clearFrame() noexcept
{
  last_match_count_ = 0;
  last_matched_ids_.clear();
  // 每帧先把两个槽的显示清单清空：后面只有真正完成滤波更新的槽会重新填，
  // 早退、初始化和 TempLost 纯预测都不会泄漏上一帧的结果。
  active_.used_lights.clear();
  backup_.used_lights.clear();
}

void EskfTracker::dropOnGap(TimePoint timestamp)
{
  if (!last_frame_) {
    return;
  }
  const bool continuous =
    timestamp >= *last_frame_ &&
    elapsedSeconds(*last_frame_, timestamp) <= tracker_config_.max_frame_gap;
  if (continuous) {
    return;
  }

  const bool had_target = active_.lifecycle.state != TrackState::Lost;
  active_.lifecycle.reset();
  backup_.lifecycle.reset();
  if (had_target) {
    ++drop_count_;
    L6Telemetry::logWarn(
      "EskfTracker: 帧间隔过长，目标复位",
      std::chrono::duration<double>(timestamp - *last_frame_).count());
  }
}

bool EskfTracker::advance(Slot & slot, Frame & frame, std::optional<ArmorName> prefer)
{
  const bool found = (slot.lifecycle.state == TrackState::Lost)
                       ? initTarget(slot, initCandidates(frame), frame, prefer)
                       : updateTarget(slot, frame);

  updateFsm(
    found, slot.lifecycle, tracker_config_.tracking_thres,
    elapsedSeconds(slot.last_update, frame.timestamp), lostThreshold(slot.target));

  // 发散的目标直接丢掉，别让它把下游一起带歪。
  if (slot.lifecycle.isTracking() && slot.target.diverged()) {
    slot.lifecycle.reset();
    slot.used_lights.clear();
    L6Telemetry::logWarn("EskfTracker: 目标发散，已复位");
  }
  return found;
}

void EskfTracker::runBackup(Frame & frame)
{
  if (active_.lifecycle.state == TrackState::Tracking) {
    backup_.lifecycle.reset();
    return;
  }
  if (active_.lifecycle.state != TrackState::TempLost) {
    return;
  }

  advance(backup_, frame, active_.target.name);

  // 当前目标的外推已经停住、又在别处看到了同一辆车：旧预测明显跟不上了，
  // 直接换成新建的目标。新目标保留 Detecting 状态和计数，连续关联够帧数
  // 才转 Tracking，L5 在那之前不会开火；换上来只是让输出立刻跟到板上。
  const bool holding = elapsedSeconds(active_.last_update, frame.timestamp) >
                       tracker_config_.temp_lost_predict_time;
  const bool reacquired = holding &&
                          backup_.lifecycle.state == TrackState::Detecting &&
                          backup_.target.name == active_.target.name;
  if (backup_.lifecycle.state == TrackState::Tracking || reacquired) {
    std::swap(active_, backup_);
    backup_.lifecycle.reset();
  }
}

// --- 单个槽位：初始化与更新 ------------------------------------------------

bool EskfTracker::initTarget(
  Slot & slot, const std::vector<Armor> & candidates, const Frame & frame,
  std::optional<ArmorName> prefer)
{
  slot.used_lights.clear();
  if (candidates.empty()) {
    return false;
  }
  // candidates 已按离图像中心的距离排过序。同一阵营里一个编号只对应一辆车，
  // 所以有同类别的板时它就是暂丢的那辆车，优先拿它重建。
  auto selected = candidates.begin();
  if (prefer) {
    const auto same = std::find_if(candidates.begin(), candidates.end(), [&](const Armor & armor) {
      return armor.name == *prefer;
    });
    if (same != candidates.end()) {
      selected = same;
    }
  }
  slot.target.reset(*selected, target_config_, frame.timestamp);
  slot.lifecycle.state = TrackState::Detecting;
  slot.lifecycle.detect_count = 0;
  slot.last_update = frame.timestamp;
  return true;
}

bool EskfTracker::updateTarget(Slot & slot, const Frame & frame)
{
  slot.used_lights.clear();

  // 先按类别筛：只有同类别的板才可能属于同一辆车。
  std::vector<Armor> same_name;
  same_name.reserve(frame.armors.size());
  for (const auto & armor : frame.armors) {
    if (armor.name == slot.target.name && semanticUsable(armor)) {
      same_name.push_back(armor);
    }
  }

  slot.target.predictEkf(frame.timestamp, holdFrom(slot));

  // 关联的两步共用同一份外推状态和同一个观测上下文。各推各的不是恒等：
  // Motion 在 dt=0 时也会跑 clamp。
  const ObsContext ctx = slot.target.obsContext(calibration_, frame.camera_in_world);
  const Eigen::VectorXd state = slot.target.stateAt(frame.timestamp);

  const auto matched = matchArmor(slot.target, state, ctx, same_name);
  const auto matched_lights =
    matchLight(slot.target, state, ctx, *frame.lights, matched, &light_match_stats_);

  // 单块完整板时补一维 IPPE 给的左右灯条深度差，这是本帧唯一会跑 PnP 的地方。
  std::optional<double> depth_difference;
  if (matched.size() == 1) {
    depth_difference = pnp_solver_.lights_depth_diff(matched.front().second);
  }

  last_match_count_ = static_cast<int>(matched.size());
  last_matched_ids_.clear();
  for (const auto & [id, armor] : matched) {
    if (!last_matched_ids_.empty()) {
      last_matched_ids_ += '|';
    }
    last_matched_ids_ += std::to_string(id);
  }

  const int updated =
    slot.target.update(matched, matched_lights, depth_difference, frame.timestamp, ctx);
  if (updated <= 0) {
    return false;
  }

  // update() 成功之后才填这份清单，它因此严格等于本帧真正进了滤波更新的那些
  // 灯条观测，而不是 matchLight 的候选或关联中间结果。
  slot.used_lights.reserve(matched.size() * 2 + matched_lights.size());
  for (const auto & [id, armor] : matched) {
    slot.used_lights.push_back({armor.points[0], armor.points[3], id, true, false});
    slot.used_lights.push_back({armor.points[1], armor.points[2], id, false, false});
  }
  for (const auto & [id, is_left, light] : matched_lights) {
    slot.used_lights.push_back({light.top, light.bottom, id, is_left, true, light.id});
  }
  slot.last_update = frame.timestamp;
  return true;
}

// --- 初始化候选 -----------------------------------------------------------

const std::vector<Armor> & EskfTracker::initCandidates(Frame & frame)
{
  if (frame.init_candidates) {
    return *frame.init_candidates;
  }

  std::vector<Armor> result;
  result.reserve(frame.armors.size());
  for (const Armor & observation : frame.armors) {
    if (!semanticUsable(observation)) {
      continue;
    }
    Armor candidate = observation;
    pnp_solver_.single_pnp(candidate);
    if (pnpUsable(candidate)) {
      result.push_back(std::move(candidate));
    }
  }
  // 按离图像中心的距离排序：初始化时优先取最近的，那通常是操作手正对的目标。
  std::sort(result.begin(), result.end(), [this](const Armor & lhs, const Armor & rhs) {
    return L6Telemetry::squaredDistance(lhs.center, image_center_) <
           L6Telemetry::squaredDistance(rhs.center, image_center_);
  });

  frame.init_candidates = std::move(result);
  return *frame.init_candidates;
}

bool EskfTracker::semanticUsable(const Armor & armor) const noexcept
{
  return armor.name != ArmorName::Unknown &&
         armor.color != L2Perception::ArmorColor::Unknown &&
         std::all_of(
           armor.points.begin(), armor.points.end(), [](const cv::Point2f & point) {
             return std::isfinite(point.x) && std::isfinite(point.y);
           });
}

bool EskfTracker::pnpUsable(const Armor & armor) const noexcept
{
  return armor.name != ArmorName::Unknown && armor.xyz_in_camera.allFinite() &&
         armor.xyz_in_camera.z() > 0.0 && armor.xyz_in_world.allFinite() &&
         armor.ypr_in_camera.allFinite() && std::isfinite(armor.reprojection_error);
}

// --- ROI ------------------------------------------------------------------

std::optional<Roi::Focus> EskfTracker::roiFocus(
  const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
  bool require_light_measurements) const
{
  if (!q_world_barrel) {
    return std::nullopt;
  }
  if (
    !active_.lifecycle.isTracking() || !active_.target.initialized() ||
    elapsedSeconds(active_.last_update, timestamp) >= lostThreshold(active_.target)) {
    return std::nullopt;
  }
  // 独立灯条 ROI 多两个条件：开关打开，且目标不是基地——基地的板不绕转，
  // 整车预测约束不了它的灯条位置。
  if (
    require_light_measurements &&
    (!active_.target.lightsEnabled() || VehicleModel::isBase(active_.target.name))) {
    return std::nullopt;
  }

  Roi::Focus focus;
  focus.target = &active_.target;
  focus.ctx = active_.target.obsContext(
    calibration_, cameraInWorld(calibration_, *q_world_barrel));
  focus.motion_end = std::min(timestamp, holdFrom(active_));
  focus.lost_time = elapsedSeconds(active_.last_update, timestamp);
  focus.lost_thres = lostThreshold(active_.target);
  return focus;
}

std::optional<cv::Rect> EskfTracker::lightRoi(
  const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
  const cv::Size & image_size) const
{
  const auto focus = roiFocus(q_world_barrel, timestamp, true);
  if (!focus) {
    return std::nullopt;
  }
  const auto box = Roi::bounds(*focus, image_size);
  if (!box) {
    return std::nullopt;
  }
  return Roi::light(*box, image_size);
}

cv::Rect EskfTracker::netFocusRoi(
  const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
  const cv::Size & image_size, double target_wh_ratio) const
{
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  const auto focus = roiFocus(q_world_barrel, timestamp, false);
  if (!focus) {
    return image_rect;
  }
  const auto box = Roi::bounds(*focus, image_size);
  if (!box) {
    return image_rect;
  }
  return Roi::net(*focus, *box, image_size, target_wh_ratio);
}

// --- 杂项 -----------------------------------------------------------------

double EskfTracker::lostThreshold(const EskfTarget & target) const noexcept
{
  return target.name == ArmorName::Outpost ? tracker_config_.lost_time_thres_outpost
                                           : tracker_config_.lost_time_thres;
}

TimePoint EskfTracker::holdFrom(const Slot & slot) const noexcept
{
  return slot.last_update + std::chrono::duration_cast<TimePoint::duration>(
                              std::chrono::duration<double>(
                                tracker_config_.temp_lost_predict_time));
}

std::vector<Eigen::Vector4d> EskfTracker::armorPoses() const
{
  if (active_.lifecycle.state == TrackState::Lost || !active_.target.initialized()) {
    return {};
  }
  return active_.target.armor_xyza_list();
}

}  // namespace L3Estimation
