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
  active_ = TrackSlot{};
  backup_ = TrackSlot{};
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
  // 早退路径也要先清，免得下游读到隔帧的采纳清单。
  active_.clearFrame();
  backup_.clearFrame();
  if (!ready_ || !q_world_barrel) {
    LOG_WARN("EskfTracker: 未就绪或缺少枪管姿态，无法跟踪");
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

  const bool was_alive = active_.alive();
  advance(active_, frame, std::nullopt);
  runBackup(frame);

  // 当前槽本帧刚退回 Lost（Detecting 丢帧、TempLost 超时或发散）：就地拿本帧
  // 的检测重建，不空等一帧。备用槽的任务随之结束，免得留着一个过期的
  // Detecting 目标。
  if (was_alive && !active_.alive()) {
    ++drop_count_;
    backup_.drop();
    advance(active_, frame, std::nullopt);
  }

  if (!active_.alive()) {
    return std::nullopt;
  }
  // 返回不含滤波器的副本，下游随便外推都不会污染滤波器状态。
  return active_.target().snapshot();
}

// 单个槽：建目标与更新
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

  const bool had_target = active_.alive();
  active_.drop();
  backup_.drop();
  if (had_target) {
    ++drop_count_;
    LOG_WARN(
      "EskfTracker: 帧间隔过长，目标复位",
      std::chrono::duration<double>(timestamp - *last_frame_).count());
  }
}

/// --- 单个槽：建目标与更新 ---
void EskfTracker::advance(TrackSlot & slot, Frame & frame, std::optional<ArmorName> prefer)
{
  const bool found =
    slot.alive() ? updateTarget(slot, frame) : initTarget(slot, frame, prefer);
  slot.step(found, frame.timestamp, tracker_config_.tracking_thres, lostThreshold(slot.name()));
}

void EskfTracker::runBackup(Frame & frame)
{
  if (active_.state() == TrackState::Tracking) {
    backup_.drop();
    return;
  }
  if (active_.state() != TrackState::TempLost) {
    return;
  }

  advance(backup_, frame, active_.name());

  // 当前目标的外推已经停住、又在别处看到了同一辆车：旧预测明显跟不上了，
  // 直接换成新建的目标。新目标保留 Detecting 状态和计数，连续关联够帧数
  // 才转 Tracking，L5 在那之前不会开火；换上来只是让输出立刻跟到板上。
  const bool holding = active_.idle(frame.timestamp) > tracker_config_.temp_lost_predict_time;
  const bool reacquired = holding && backup_.state() == TrackState::Detecting &&
                          backup_.name() == active_.name();
  if (backup_.state() == TrackState::Tracking || reacquired) {
    std::swap(active_, backup_);
    backup_.drop();
  }
}

// --- 单个槽：建目标与更新 --------------------------------------------------

bool EskfTracker::initTarget(TrackSlot & slot, Frame & frame, std::optional<ArmorName> prefer)
{
  const std::vector<Armor> & candidates = initCandidates(frame);
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
  slot.start(*selected, target_config_, frame.timestamp);
  return true;
}

bool EskfTracker::updateTarget(TrackSlot & slot, const Frame & frame)
{
  slot.predict(frame.timestamp, tracker_config_.temp_lost_predict_time);
  const EskfTarget & target = slot.target();

  // 先按类别筛：只有同类别的板才可能属于同一辆车。
  std::vector<Armor> same_name;
  same_name.reserve(frame.armors.size());
  for (const auto & armor : frame.armors) {
    if (armor.name == target.name && semanticUsable(armor)) {
      same_name.push_back(armor);
    }
  }

  // 关联的两步共用同一份外推状态和同一个观测上下文。各推各的不是恒等：
  // Motion 在 dt=0 时也会跑 clamp。
  const ObsContext ctx = target.obsContext(calibration_, frame.camera_in_world);
  const Eigen::VectorXd state = target.stateAt(frame.timestamp);

  const auto matched = matchArmor(target, state, ctx, same_name);
  const auto matched_lights =
    matchLight(target, state, ctx, *frame.lights, matched, &light_match_stats_);

  // 单块完整板时补一维 IPPE 给的左右灯条深度差，这是本帧唯一会跑 PnP 的地方。
  std::optional<double> depth_difference;
  if (matched.size() == 1) {
    depth_difference = pnp_solver_.lights_depth_diff(matched.front().second);
  }

  return slot.update(matched, matched_lights, depth_difference, frame.timestamp, ctx);
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
  bool for_lights) const
{
  if (!q_world_barrel) {
    return std::nullopt;
  }
  // init_roi 时网络 ROI 从 Detecting 起就给：初始化用的已经是 ROI 检测，
  // 接下来几帧若退回整图，尺度又切一次，等于没做。
  const bool live = (for_lights || !tracker_config_.init_roi) ? active_.confirmed()
                                                              : active_.alive();
  if (!live || active_.idle(timestamp) >= lostThreshold(active_.name())) {
    return std::nullopt;
  }
  // 独立灯条 ROI 多两个条件：开关打开，且目标不是基地——基地的板不绕转，
  // 整车预测约束不了它的灯条位置。
  if (
    for_lights &&
    (!active_.target().lightsEnabled() || VehicleModel::isBase(active_.name()))) {
    return std::nullopt;
  }

  return focusOn(
    active_.target(), cameraInWorld(calibration_, *q_world_barrel),
    std::min(timestamp, active_.holdFrom(tracker_config_.temp_lost_predict_time)),
    active_.idle(timestamp));
}

Roi::Focus EskfTracker::focusOn(
  const EskfTarget & target, const Eigen::Isometry3d & camera_in_world, TimePoint motion_end,
  double lost_time) const
{
  Roi::Focus focus;
  focus.target = &target;
  focus.ctx = target.obsContext(calibration_, camera_in_world);
  focus.motion_end = motion_end;
  focus.lost_time = lost_time;
  focus.lost_thres = lostThreshold(target.name);
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

std::vector<L2Perception::LightHint> EskfTracker::lightHints(
  const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp) const
{
  const auto focus = roiFocus(q_world_barrel, timestamp, true);
  return focus ? Roi::hints(*focus) : std::vector<L2Perception::LightHint>{};
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

std::optional<cv::Rect> EskfTracker::initRoi(
  const std::vector<L2Perception::Armor> & detections,
  const std::optional<Eigen::Quaterniond> & q_world_barrel, TimePoint timestamp,
  const cv::Size & image_size, double target_wh_ratio)
{
  if (!ready_ || !tracker_config_.init_roi || !q_world_barrel || active_.alive()) {
    return std::nullopt;
  }
  // 和 track() 里的初始化同一套挑法：同一份候选、同一个排序、取第一块。
  pnp_solver_.set_R_world_barrel(q_world_barrel);
  Frame frame{
    .timestamp = timestamp,
    .camera_in_world = cameraInWorld(calibration_, *q_world_barrel),
    .armors = toObservations(detections)};
  const std::vector<Armor> & candidates = initCandidates(frame);
  if (candidates.empty()) {
    return std::nullopt;
  }

  // 临时目标只用来投影整车包围盒，窗口大小因此和跟踪时的 Roi::net 一致。
  // 整图和 ROI 的尺度差来自两者放大倍率不同，所以初始化这一窗口也要按跟踪
  // 时的算法定大小，而不是随手在板周围框一块。
  EskfTarget probe;
  probe.reset(candidates.front(), target_config_, timestamp);
  const Roi::Focus focus = focusOn(probe, frame.camera_in_world, timestamp, 0.0);
  const auto box = Roi::bounds(focus, image_size);
  if (!box) {
    return std::nullopt;
  }
  return Roi::net(focus, *box, image_size, target_wh_ratio);
}

// --- 杂项 -----------------------------------------------------------------

// 每块完整板在采纳清单里恰好拆成一左一右两根非独立灯条，按左灯条数就是板数，
// 顺序就是关联的顺序。
int EskfTracker::lastMatchCount() const noexcept
{
  const auto & used = active_.usedLights();
  return static_cast<int>(std::count_if(used.begin(), used.end(), [](const UsedLight & light) {
    return !light.isolated && light.is_left;
  }));
}

std::string EskfTracker::lastMatchedIds() const
{
  std::string ids;
  for (const UsedLight & light : active_.usedLights()) {
    if (light.isolated || !light.is_left) {
      continue;
    }
    if (!ids.empty()) {
      ids += '|';
    }
    ids += std::to_string(light.armor_id);
  }
  return ids;
}

double EskfTracker::lostThreshold(ArmorName name) const noexcept
{
  return name == ArmorName::Outpost ? tracker_config_.lost_time_thres_outpost
                                    : tracker_config_.lost_time_thres;
}

std::vector<Eigen::Vector4d> EskfTracker::armorPoses() const
{
  if (!active_.alive()) {
    return {};
  }
  return active_.target().armor_xyza_list();
}

}  // namespace L3Estimation
