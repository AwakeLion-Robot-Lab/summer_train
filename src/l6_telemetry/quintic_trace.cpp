#include "l6_telemetry/quintic_trace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace L6Telemetry {
namespace {
using namespace L4Planning;
using Json = nlohmann::json;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double delta(double a, double b)
{
  return std::remainder(a - b, 2.0 * std::numbers::pi);
}

AimState invalid()
{
  return {{kNaN, kNaN, kNaN}, {kNaN, kNaN, kNaN}};
}

bool finite(const AxisState& axis)
{
  return std::isfinite(axis.position) && std::isfinite(axis.velocity) &&
    std::isfinite(axis.acceleration);
}

AimState evaluate(const BlendSolution& segment, double t)
{
  return {{segment.yaw.position(t), segment.yaw.velocity(t), segment.yaw.acceleration(t)},
          {segment.pitch.position(t), segment.pitch.velocity(t), segment.pitch.acceleration(t)}};
}

Json difference(const AimState& a, const AimState& b, bool wrap_angles = true)
{
  const auto position = [wrap_angles](double x, double y) {
    return wrap_angles ? delta(x, y) : x - y;
  };
  return {
    {"yaw_position_rad", position(a.yaw.position, b.yaw.position)},
    {"pitch_position_rad", position(a.pitch.position, b.pitch.position)},
    {"yaw_velocity_rad_s", a.yaw.velocity - b.yaw.velocity},
    {"pitch_velocity_rad_s", a.pitch.velocity - b.pitch.velocity},
    {"yaw_acceleration_rad_s2", a.yaw.acceleration - b.yaw.acceleration},
    {"pitch_acceleration_rad_s2", a.pitch.acceleration - b.pitch.acceleration}};
}

bool segmentFinite(const BlendSolution& segment)
{
  if (!segment.valid || !std::isfinite(segment.duration) || segment.duration <= 0.0) {
    return false;
  }
  for (int i = 0; i < 6; ++i) {
    if (!std::isfinite(segment.yaw.coefficient[i]) ||
        !std::isfinite(segment.pitch.coefficient[i])) {
      return false;
    }
  }
  return finite(evaluate(segment, 0.0).yaw) &&
    finite(evaluate(segment, segment.duration).yaw) &&
    finite(evaluate(segment, 0.0).pitch) &&
    finite(evaluate(segment, segment.duration).pitch);
}
}  // namespace

QuinticTrace::QuinticTrace(BlendLimits limits, double yaw_speed, double pitch_speed)
: limits_(limits), yaw_speed_(yaw_speed), pitch_speed_(pitch_speed)
{
  if (!(std::isfinite(yaw_speed) && yaw_speed > 0.0 &&
        std::isfinite(pitch_speed) && pitch_speed > 0.0)) {
    throw std::invalid_argument("diagnostic speed limits must be finite and positive");
  }
}

AxisState QuinticTrace::Difference::update(double angle, double step)
{
  AxisState output{angle, kNaN, kNaN};
  if (!std::isfinite(angle)) {
    *this = {};
    return output;
  }
  if (has_position && step > 0.0) {
    output.velocity = delta(angle, position) / step;
    // 两段割线速度位于各自区间中点；适用于不等间隔图像帧。
    if (has_velocity) {
      output.acceleration = 2.0 * (output.velocity - velocity) / (step + dt);
    }
    velocity = output.velocity;
    has_velocity = true;
  }
  position = angle;
  dt = step;
  has_position = true;
  return output;
}

void QuinticTrace::reset()
{
  previous_valid_ = false;
  previous_reference_ = {};
  committed_after_ = {};
  previous_segment_ = {};
  raw_yaw_ = raw_pitch_ = planned_yaw_ = planned_pitch_ = {};
  ++resets_;
}

Json QuinticTrace::update(
  TimePoint now, const Plan& plan, const PlannerDiagnostics& d,
  double measured_yaw, double measured_pitch)
{
  if (!started_) {
    origin_ = previous_time_ = now;
    started_ = true;
  }
  const double dt = std::chrono::duration<double>(now - previous_time_).count();
  if (dt < 0.0) {
    throw std::invalid_argument("quintic telemetry requires monotonic time");
  }
  const bool raw_valid = d.raw_valid && plan.valid() && plan.fire.has_value();
  AimState raw = raw_valid ? d.raw : invalid();
  // 角度展示真实下发前的射击轨迹；导数仍来自同板快照。
  if (raw_valid) {
    raw.yaw.position = plan.aim.shootYaw();
    raw.pitch.position = plan.aim.shootPitch();
  }
  const bool segment_valid = segmentFinite(d.segment);
  const double tau = std::chrono::duration<double>(now - d.segment_start).count();
  AimState planned = invalid();
  if (plan.valid()) {
    planned = plan.aim.blending && segment_valid
      ? evaluate(d.segment, std::clamp(tau, 0.0, d.segment.duration)) : raw;
    planned.yaw.position = plan.aim.yaw;
    planned.pitch.position = plan.aim.pitch;
  }
  const AimState old_reference = previous_reference_
    ? previous_reference_(dt) : invalid();
  const double old_tau = std::chrono::duration<double>(
    now - previous_segment_start_).count();
  const bool ended = previous_segment_.valid && old_tau >= previous_segment_.duration;
  // 在同一终点时刻比较旧多项式和最新射击轨迹，不把帧间正常运动当成跳变。
  const AimState join_left = ended
    ? evaluate(previous_segment_, previous_segment_.duration) : invalid();
  const AimState join_right = ended && d.raw_trajectory
    ? d.raw_trajectory(previous_segment_.duration - old_tau) : invalid();
  if (previous_valid_ && dt > 0.0) {
    observed_seconds_ += dt;
    if (previous_segment_.valid) {
      const double from = std::chrono::duration<double>(
        previous_time_ - previous_segment_start_).count();
      blend_seconds_ += std::max(0.0,
        std::min(old_tau, previous_segment_.duration) - std::max(from, 0.0));
    }
  }
  if (d.smoother.committed) {
    ++commits_;
    late_commits_ += d.smoother.late ? 1 : 0;
    infeasible_commits_ += d.segment.acceleration_limited ? 1 : 0;
    committed_after_ = d.forecast ? d.forecast->after : TrajectorySampler{};
  }
  const auto raw_yaw_fd = raw_yaw_.update(raw.yaw.position, dt);
  const auto raw_pitch_fd = raw_pitch_.update(raw.pitch.position, dt);
  const auto yaw_fd = planned_yaw_.update(planned.yaw.position, dt);
  const auto pitch_fd = planned_pitch_.update(planned.pitch.position, dt);
  const auto axis = [](const AxisState& r, const AxisState& q,
                       const AxisState& rf, const AxisState& qf,
                       double measured, double max_v, double max_a,
                       double peak_v, double peak_a) -> Json {
    return {
      {"raw_rad", r.position}, {"planned_rad", q.position}, {"measured_rad", measured},
      {"raw_velocity_rad_s", r.velocity}, {"planned_velocity_rad_s", q.velocity},
      {"raw_acceleration_rad_s2", r.acceleration}, {"planned_acceleration_rad_s2", q.acceleration},
      {"raw_fd_velocity_rad_s", rf.velocity}, {"planned_fd_velocity_rad_s", qf.velocity},
      {"raw_fd_acceleration_rad_s2", rf.acceleration}, {"planned_fd_acceleration_rad_s2", qf.acceleration},
      {"error_rad", delta(q.position, r.position)}, {"max_speed_rad_s", max_v},
      {"min_speed_rad_s", -max_v}, {"max_acc_rad_s2", max_a}, {"min_acc_rad_s2", -max_a},
      {"segment_peak_speed_rad_s", peak_v}, {"segment_peak_acc_rad_s2", peak_a},
      {"segment_speed_exceeded", std::isfinite(peak_v) ? double(peak_v > max_v) : kNaN},
      {"segment_acc_exceeded", std::isfinite(peak_a) ? double(peak_a > max_a) : kNaN}};
  };
  const double yaw_peak_v = segment_valid ? d.segment.yaw.peakAbsVelocity() : kNaN;
  const double pitch_peak_v = segment_valid ? d.segment.pitch.peakAbsVelocity() : kNaN;
  const bool candidate_valid = segmentFinite(d.smoother.candidate);
  const auto& candidate = d.smoother.candidate;
  Json result{
    {"t", std::chrono::duration<double>(now - origin_).count()},
    {"dt_s", dt}, {"valid", int(plan.valid())}, {"raw_valid", int(raw_valid)},
    {"raw_derivatives_valid", int(raw_valid && finite(d.raw.yaw) && finite(d.raw.pitch))},
    {"raw_sampler_error", {
      {"yaw_rad", raw_valid ? delta(d.raw.yaw.position, raw.yaw.position) : kNaN},
      {"pitch_rad", raw_valid ? delta(d.raw.pitch.position, raw.pitch.position) : kNaN}}},
    {"reason", static_cast<int>(plan.reason)},
    {"armor_id", plan.fire ? plan.fire->armor_id : -1}, {"next_armor_id", d.next_armor_id},
    {"yaw", axis(raw.yaw, planned.yaw, raw_yaw_fd, yaw_fd, measured_yaw,
      yaw_speed_, limits_.max_yaw_acceleration, yaw_peak_v,
      segment_valid ? d.segment.yaw.peakAbsAcceleration() : kNaN)},
    {"pitch", axis(raw.pitch, planned.pitch, raw_pitch_fd, pitch_fd, measured_pitch,
      pitch_speed_, limits_.max_pitch_acceleration, pitch_peak_v,
      segment_valid ? d.segment.pitch.peakAbsAcceleration() : kNaN)},
    {"blend", {
      {"active", int(plan.aim.blending)}, {"committed", int(d.smoother.committed)},
      {"progress", d.smoother.progress}, {"late", int(d.smoother.late)},
      {"duration_ms", segment_valid ? d.segment.duration * 1000.0 : kNaN},
      {"forecast_ms", d.forecast ? d.forecast->switch_time * 1000.0 : kNaN},
      {"segment_valid", int(segment_valid)},
      {"segment_acc_feasible", segment_valid ? double(!d.segment.acceleration_limited) : kNaN},
      {"search_attempted", int(d.smoother.search_attempted)},
      {"search_valid", d.smoother.search_attempted ? double(candidate_valid) : kNaN},
      {"search_acc_feasible", d.smoother.search_attempted
        ? double(candidate_valid && !candidate.acceleration_limited) : kNaN},
      {"candidate_duration_ms", candidate_valid ? candidate.duration * 1000.0 : kNaN},
      {"candidate_peak_yaw_acc", candidate_valid ? candidate.peak_yaw_acceleration : kNaN},
      {"candidate_peak_pitch_acc", candidate_valid ? candidate.peak_pitch_acceleration : kNaN},
      {"speed_limit_enforced", 0}}},
    {"boundary", {
      {"valid", int(segment_valid)},
      {"start", difference(segment_valid ? evaluate(d.segment, 0.0) : invalid(),
                           segment_valid ? d.segment.start : invalid(), false)},
      {"end", difference(segment_valid ? evaluate(d.segment, d.segment.duration) : invalid(),
                         segment_valid ? d.segment.end : invalid(), false)}}},
    {"replan", difference(planned, old_reference)},
    {"join", {{"end_event", int(ended)},
      {"end_valid", int(ended && d.raw_trajectory && finite(join_right.yaw) && finite(join_right.pitch))},
      {"end", difference(join_right, join_left)}}},
    {"counts", {{"commits", commits_}, {"late_commits", late_commits_},
      {"acc_infeasible_commits", infeasible_commits_}, {"resets", resets_}}},
    {"coverage", {{"observed_s", observed_seconds_}, {"blend_s", blend_seconds_},
      {"blend_fraction", observed_seconds_ > 0.0 ? blend_seconds_ / observed_seconds_ : 0.0}}}};

  if (plan.valid() && plan.aim.blending && segment_valid) {
    // 保存本帧完整参考模型，下一帧在同一时刻求值，再量重新规划的偏差。
    previous_reference_ = [segment = d.segment, tau, after = committed_after_](double offset) {
      const double t = tau + offset;
      if (t <= segment.duration) {
        return evaluate(segment, t);
      }
      return after ? after(t) : invalid();
    };
  } else {
    previous_reference_ = raw_valid ? d.raw_trajectory : TrajectorySampler{};
  }
  previous_segment_ = plan.valid() && plan.aim.blending ? d.segment : BlendSolution{};
  previous_segment_start_ = d.segment_start;
  previous_time_ = now;
  previous_valid_ = plan.valid();
  return result;
}
}  // namespace L6Telemetry
