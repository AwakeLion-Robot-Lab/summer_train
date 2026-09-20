#include "l3_estimation/armor/eskf_tracker.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <chrono>
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
         config.lost_time_thres_outpost >= config.lost_time_thres &&
         config.max_frame_gap > 0.0 && config.temp_lost_predict_time > 0.0;
}

}  // namespace

Armor toObservation(
  const L2Perception::Armor& detection, TimePoint timestamp)
{
  // 只搬检测字段，三维位姿留给当帧的 PnpSolver 填。
  Armor observation;
  observation.class_id = detection.class_id;
  observation.color = detection.color;
  observation.points = detection.corners;
  observation.center = detection.center;
  observation.confidence = static_cast<double>(detection.confidence);
  observation.area = L6Telemetry::polygonArea(detection.corners);
  observation.timestamp = timestamp;
  return observation;
}

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
  // 板尺寸只认这一份，免得 PnP 物点和灯条端点观测用上两套几何。
  target_config_.armor = armor_config_;
}

void EskfTracker::reset() noexcept
{
  for (auto & slot : buffer_) {
    slot.lifecycle.reset();
    slot.target = EskfTarget{};
    slot.used_lights.clear();
  }
  current_ = 0;
  previous_ = 1;
  last_frame_.reset();
  observations_.clear();
}

bool EskfTracker::semanticUsable(const Armor& armor) const noexcept
{
  return armor.name != ArmorName::Unknown &&
         armor.color != L2Perception::ArmorColor::Unknown &&
         std::all_of(
           armor.points.begin(), armor.points.end(),
           [](const cv::Point2f& point) {
             return std::isfinite(point.x) && std::isfinite(point.y);
           });
}

bool EskfTracker::pnpUsable(const Armor& armor) const noexcept
{
  return armor.name != ArmorName::Unknown && armor.xyz_in_camera.allFinite() &&
         armor.xyz_in_camera.z() > 0.0 && armor.xyz_in_world.allFinite() &&
         armor.ypr_in_camera.allFinite() &&
         std::isfinite(armor.reprojection_error);
}

std::vector<Armor> EskfTracker::initCandidates()
{
  std::vector<Armor> result;
  result.reserve(observations_.size());
  for (const Armor& observation : observations_) {
    if (!semanticUsable(observation)) {
      continue;
    }
    Armor pnp_observation = observation;
    pnp_solver_.single_pnp(pnp_observation);
    if (pnpUsable(pnp_observation)) {
      result.push_back(std::move(pnp_observation));
    }
  }
  // 按离图像中心的距离排序：初始化时优先取最近的，那通常是操作手正对的目标。
  std::sort(result.begin(), result.end(), [this](const Armor& lhs, const Armor& rhs) {
    return L6Telemetry::squaredDistance(lhs.center, image_center_) <
           L6Telemetry::squaredDistance(rhs.center, image_center_);
  });
  return result;
}

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

bool EskfTracker::initTarget(
  Slot& slot, const std::vector<Armor>& candidates, TimePoint timestamp,
  const Eigen::Isometry3d & camera_in_world, std::optional<ArmorName> prefer)
{
  slot.used_lights.clear();
  if (candidates.empty()) {
    return false;
  }
  // candidates 已按离图像中心的距离排过序。同一阵营里一个编号只对应一辆车，
  // 所以有同类别的板时它就是暂丢的那辆车，优先拿它重建。
  auto selected_it = candidates.begin();
  if (prefer) {
    const auto same = std::find_if(candidates.begin(), candidates.end(), [&](const Armor & armor) {
      return armor.name == *prefer;
    });
    if (same != candidates.end()) {
      selected_it = same;
    }
  }
  const Armor& selected = *selected_it;
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
  slot.used_lights.clear();

  // 先按类别筛：只有同类别的板才可能属于同一辆车。
  std::vector<Armor> same_name;
  same_name.reserve(candidates.size());
  for (const auto & armor : candidates) {
    if (armor.name == slot.target.name && semanticUsable(armor)) {
      same_name.push_back(armor);
    }
  }

  slot.target.predictEkf(timestamp, holdFrom(slot));
  const auto matched =
    slot.target.matchArmor(same_name, timestamp, calibration_, camera_in_world);
  const auto matched_lights = slot.target.matchLight(
    lights, matched, timestamp, calibration_, camera_in_world,
    &light_match_stats_);

  std::optional<double> depth_difference;
  if (matched.size() == 1) {
    depth_difference = pnp_solver_.lights_depth_diff(
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
    // update() 成功之后才填这份清单，它因此严格等于本帧真正进了 updateMulti
    // 的那些灯条观测，而不是 matchLight 的候选或关联中间结果。
    slot.used_lights.reserve(
      matched.size() * 2 + matched_lights.size());
    for (const auto& [id, armor] : matched) {
      slot.used_lights.push_back(
        {armor.points[0], armor.points[3], id, true, false});
      slot.used_lights.push_back(
        {armor.points[1], armor.points[2], id, false, false});
    }
    for (const auto& [id, is_left, light] : matched_lights) {
      slot.used_lights.push_back(
        {light.top, light.bottom, id, is_left, true, light.id});
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
  // 每帧先把两个槽的显示清单清空：后面只有真正完成滤波更新的槽会重新填，
  // 早退、初始化和 TempLost 纯预测都不会泄漏上一帧的结果。
  for (auto& slot : buffer_) {
    slot.used_lights.clear();
  }
  if (!ready_ || !q_world_barrel) {
    return std::nullopt;
  }

  // 断流（间隔过长或时间倒退）后旧目标的外推不可信，两个槽都清掉，这一帧按
  // Lost 重新挑候选初始化。
  if (last_frame_ &&
      (timestamp < *last_frame_ ||
       elapsedSeconds(*last_frame_, timestamp) > tracker_config_.max_frame_gap)) {
    const bool had_target = buffer_[current_].lifecycle.state != TrackState::Lost;
    for (auto & slot : buffer_) {
      slot.lifecycle.reset();
    }
    if (had_target) {
      ++drop_count_;
      L6Telemetry::logWarn(
        "EskfTracker: 帧间隔过长，目标复位",
        std::chrono::duration<double>(timestamp - *last_frame_).count());
    }
  }
  last_frame_ = timestamp;

  // L2 -> L3：正常更新只搬类别和四角点，不拿 PnP 成功当入口门限。PnP 只在
  // Lost 初始化和单块完整板求深度差时才跑。
  pnp_solver_.set_R_world_barrel(q_world_barrel);
  observations_.reserve(detections.size());
  for (const auto & detection : detections) {
    Armor observation = toObservation(detection, timestamp);
    observation.name = L2Perception::armorClassFromId(observation.class_id);
    if (const auto type = armorTypeOf(observation.name)) {
      observation.type = *type;
    }
    observations_.push_back(std::move(observation));
  }

  const Eigen::Isometry3d camera_in_world =
    EskfTarget::cameraInWorld(calibration_, *q_world_barrel);

  // 初始化候选按需算、本帧内缓存：正常 Tracking 更新一次 PnP 都不跑；当前槽
  // 恰好在这一帧转进 TempLost 时，紧接着处理备用槽还能当帧完成初始化，不用
  // 白等一帧。
  std::optional<std::vector<Armor>> initialization_candidates;
  const auto getInitializationCandidates = [&]() -> const std::vector<Armor>& {
    if (!initialization_candidates) {
      initialization_candidates = initCandidates();
    }
    return *initialization_candidates;
  };

  const auto process = [&](std::size_t index, std::optional<ArmorName> prefer) {
    Slot & slot = buffer_[index];
    const bool found = (slot.lifecycle.state == TrackState::Lost)
                         ? initTarget(
                             slot, getInitializationCandidates(), timestamp,
                             camera_in_world, prefer)
                         : updateTarget(
                             slot, observations_, lights, timestamp,
                             camera_in_world);

    updateFsm(
      found, slot.lifecycle, tracker_config_.tracking_thres,
      elapsedSeconds(slot.last_update, timestamp), lostThreshold(slot.target));

    // 发散的目标直接丢掉，别让它把下游一起带歪。
    if (slot.lifecycle.isTracking() && slot.target.diverged()) {
      slot.lifecycle.reset();
      slot.used_lights.clear();
      L6Telemetry::logWarn("EskfTracker: 目标发散，已复位");
    }
    return found;
  };

  // 双缓冲：当前目标进 TempLost 时让另一个槽同时抓新目标，新目标一转成
  // Tracking 就交换上来，不必等当前目标超时。
  Slot & current = buffer_[current_];
  Slot & previous = buffer_[previous_];

  const bool was_active = current.lifecycle.state != TrackState::Lost;
  process(current_, std::nullopt);

  if (current.lifecycle.state == TrackState::TempLost) {
    process(previous_, current.target.name);
    // 当前目标的外推已经停住、又在别处看到了同一辆车：旧预测明显跟不上了，
    // 直接换成新建的目标。新目标保留 Detecting 状态和计数，连续关联够帧数
    // 才转 Tracking，L5 在那之前不会开火；换上来只是让输出立刻跟到板上。
    const bool holding =
      elapsedSeconds(current.last_update, timestamp) > tracker_config_.temp_lost_predict_time;
    const bool reacquired = holding &&
                            previous.lifecycle.state == TrackState::Detecting &&
                            previous.target.name == current.target.name;
    if (previous.lifecycle.state == TrackState::Tracking || reacquired) {
      std::swap(current, previous);
      previous.lifecycle.reset();
    }
  } else if (current.lifecycle.state == TrackState::Tracking) {
    previous.lifecycle.reset();
  }

  // 当前槽本帧刚退回 Lost（Detecting 丢帧、TempLost 超时或发散）：就地拿本帧
  // 的检测重建，不空等一帧。备用槽的任务随之结束，免得留着一个过期的
  // Detecting 目标。
  if (was_active && current.lifecycle.state == TrackState::Lost) {
    ++drop_count_;
    previous.lifecycle.reset();
    process(current_, std::nullopt);
  }

  const Slot & active = buffer_[current_];
  if (active.lifecycle.state == TrackState::Lost || !active.target.initialized()) {
    return std::nullopt;
  }
  // 返回不含滤波器的副本，下游随便外推都不会污染滤波器状态。
  return active.target.snapshot();
}

std::optional<cv::Rect> EskfTracker::lightBounds(
  const std::optional<Eigen::Quaterniond>& q_world_barrel, TimePoint timestamp,
  const cv::Size& image_size, bool require_light_measurements) const
{
  if (!q_world_barrel || image_size.width <= 0 || image_size.height <= 0) {
    return std::nullopt;
  }
  const Slot& active = buffer_[current_];
  if (!active.lifecycle.isTracking() || !active.target.initialized() ||
      elapsedSeconds(active.last_update, timestamp) >=
        lostThreshold(active.target)) {
    return std::nullopt;
  }
  // 独立灯条 ROI 多两个条件：开关打开，且目标不是基地——基地的板不绕转，
  // 整车预测约束不了它的灯条位置。
  if (require_light_measurements &&
      (!active.target.lightsEnabled() ||
       active.target.name == ArmorName::BaseSmall ||
       active.target.name == ArmorName::BaseLarge)) {
    return std::nullopt;
  }

  // 与滤波器同一个外推截止：超过 temp_lost_predict_time 没更新就只推到截止时刻。
  EskfTarget predicted = active.target.snapshot();
  const TimePoint motion_end = std::min(timestamp, holdFrom(active));
  if (motion_end > predicted.t()) {
    predicted.predict(motion_end);
  }
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

// 保持中心不动按比例放大矩形，再裁回图像范围内。
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

std::optional<cv::Rect> EskfTracker::lightRoi(
  const std::optional<Eigen::Quaterniond>& q_world_barrel,
  TimePoint timestamp, const cv::Size& image_size) const
{
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  const auto bounds = lightBounds(q_world_barrel, timestamp, image_size, true);
  if (!bounds) {
    return std::nullopt;
  }

  // 这个 ROI 越紧越好：范围一大，别的车和环境灯光就容易混进候选，CPU 开销和
  // 误匹配概率一起上去。
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
  // 不可聚焦时返回整图而不是空，调用方拿到的永远是能直接用的矩形。
  const auto bounds = lightBounds(q_world_barrel, timestamp, image_size, false);
  if (!bounds) {
    return image_rect;
  }

  const Slot& active = buffer_[current_];
  const bool is_base = active.target.name == ArmorName::BaseSmall ||
                       active.target.name == ArmorName::BaseLarge;
  // 基地尺寸大、整车模型退化，ROI 放得更宽。
  constexpr double kExpandRatio = 1.4;
  constexpr double kExpandRatioBase = 3.0;
  cv::Rect rect =
    expandAndClip(*bounds, is_base ? kExpandRatioBase : kExpandRatio, image_rect);
  if (rect.empty()) {
    return image_rect;
  }

  // ① 按网络输入宽高比修正形状：相机图像的长宽比通常与网络输入不一致，直接
  //    letterbox 会整体缩小，先把 ROI 修成同一比例能少填不少边。
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

  // ② 按距上次更新的时长线性膨胀，超时直接退化成整图：越久没更新预测越不可
  //    信，搜索范围就该越大，跟丢时 ROI 会自动放手，而不是把网络锁死在一个
  //    错误的小窗口里。
  const double lost_time = elapsedSeconds(active.last_update, timestamp);
  const double lost_thres = lostThreshold(active.target);
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

std::vector<Eigen::Vector4d> EskfTracker::armorPoses() const
{
  const Slot & active = buffer_[current_];
  if (active.lifecycle.state == TrackState::Lost || !active.target.initialized()) {
    return {};
  }
  return active.target.armor_xyza_list();
}

}  // namespace L3Estimation
