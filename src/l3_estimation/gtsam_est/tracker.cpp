#include "l3_estimation/gtsam_est/tracker.hpp"

#include "l3_estimation/pnp_solver.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

#ifdef NEWVISION_USE_GTSAM
#include "l3_estimation/gtsam_est/target.hpp"
#endif

namespace L3Estimation::GtsamEst {
namespace {

[[nodiscard]] bool validGraphConfig(const Config& config) noexcept
{
  return config.max_match_distance > 0.0 && config.max_match_yaw_diff > 0.0 &&
         config.first_update_batch_size > 0 &&
         config.lost_threshold > std::chrono::milliseconds::zero() &&
         config.translation_prior_sigma > 0.0 && config.velocity_prior_sigma > 0.0 &&
         config.yaw_prior_sigma > 0.0 && config.vyaw_prior_sigma > 0.0 &&
         config.radius_prior_sigma > 0.0 && config.dz_prior_sigma > 0.0 &&
         config.default_radius > config.radius_min &&
         config.default_radius < config.radius_max &&
         config.radius_min > 0.0 && config.radius_max > config.radius_min &&
         config.translation_factor_sigma > 0.0 && config.velocity_factor_sigma > 0.0 &&
         config.yaw_factor_sigma > 0.0 && config.vyaw_factor_sigma > 0.0 &&
         config.obs_tangential_sigma > 0.0 && config.obs_radial_sigma > 0.0 &&
         config.obs_height_sigma > 0.0 && config.obs_yaw_sigma > 0.0 &&
         config.obs_pixel_sigma > 0.0;
}

}  // namespace

bool available() noexcept
{
#ifdef NEWVISION_USE_GTSAM
  return true;
#else
  return false;
#endif
}

class Tracker::Impl
{
public:
  Impl(
    const L1Sensor::CameraCalibration& calibration,
    ArmorConfig armor_config,
    TargetConfig target_config,
    Config gtsam_config)
  : calibration_(calibration),
    armor_config_(armor_config),
    target_config_(target_config),
    gtsam_config_(gtsam_config),
    pnp_solver_(calibration_, armor_config_),
    image_center_{
      static_cast<float>(calibration.image_size.width) * 0.5F,
      static_cast<float>(calibration.image_size.height) * 0.5F},
    ready_(pnp_solver_.ready() && validGraphConfig(gtsam_config_))
  {
  }

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] TrackState state() const noexcept { return state_; }
  [[nodiscard]] const std::vector<Armor>& observations() const noexcept
  {
    return observations_;
  }

  [[nodiscard]] std::optional<TrackedTarget> track(
    const std::vector<L2Perception::Armor>& detections,
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp)
  {
    observations_.clear();
    if (!ready_) {
      resetTracking();
      return std::nullopt;
    }

    pnp_solver_.set_R_world_barrel(q_world_barrel);
    observations_.reserve(detections.size());
    for (const auto& detection : detections) {
      observations_.push_back(toArmorObservation(detection, timestamp));
      pnp_solver_.single_pnp(observations_.back());
    }

    if (last_timestamp_) {
      const auto elapsed = timestamp - *last_timestamp_;
      if (elapsed <= TimePoint::duration::zero()) {
        L6Telemetry::logWarn("GtsamEst tracker reset after invalid frame interval");
        resetTracking();
      }
    }
    last_timestamp_ = timestamp;

    const std::vector<const Armor*> armors = usableObservations();
    const std::optional<Eigen::Isometry3d> T_world_camera =
      worldCameraTransform(q_world_barrel);
    bool found = false;
    try {
      if (state_ == TrackState::Lost) {
        found = T_world_camera && initializeTarget(armors, timestamp, *T_world_camera);
      } else {
        // 没有 IMU 姿态时没有可用 PnP，仍可推进纯运动状态；观测因子不会被添加。
        const Eigen::Isometry3d transform =
          T_world_camera.value_or(Eigen::Isometry3d::Identity());
        found = updateTarget(armors, timestamp, transform);
      }
    } catch (const std::exception& exception) {
      L6Telemetry::logError("GtsamEst optimization failed", exception.what());
      resetTracking();
      return std::nullopt;
    }

#ifdef NEWVISION_USE_GTSAM
    if (!target_) {
      return std::nullopt;
    }
    if (target_->diverged()) {
      L6Telemetry::logDebug("GtsamEst tracker reset: target diverged");
      resetTracking();
      return std::nullopt;
    }
    // JLU Target 没有 EKF 的 Detecting/按帧计数丢失门限：有匹配即
    // TRACKING，暂时没有匹配即 TEMPLOST，超时由 Config::lost_threshold 复位。
    state_ = found ? TrackState::Tracking : TrackState::TempLost;
    return target_->snapshot();
#else
    return std::nullopt;
#endif
  }

  [[nodiscard]] std::vector<Eigen::Vector4d> targetArmorPoses() const
  {
#ifdef NEWVISION_USE_GTSAM
    if (target_ && state_ != TrackState::Lost) {
      return target_->armorPoses();
    }
#endif
    return {};
  }

  void reset() noexcept
  {
    resetTracking();
    last_timestamp_.reset();
    observations_.clear();
  }

private:
  [[nodiscard]] std::optional<Eigen::Isometry3d> worldCameraTransform(
    const std::optional<Eigen::Quaterniond>& q_world_barrel) const
  {
    if (!q_world_barrel || !calibration_.T_barrel_camera ||
        !q_world_barrel->coeffs().allFinite() || q_world_barrel->squaredNorm() <= 1e-12) {
      return std::nullopt;
    }
    Eigen::Isometry3d T_world_barrel = Eigen::Isometry3d::Identity();
    T_world_barrel.linear() = q_world_barrel->normalized().toRotationMatrix();
    return T_world_barrel * *calibration_.T_barrel_camera;
  }

  [[nodiscard]] std::vector<const Armor*> usableObservations() const
  {
    std::vector<const Armor*> usable;
    usable.reserve(observations_.size());
    for (const Armor& observation : observations_) {
      if (observation.name != ArmorName::Unknown) {
        usable.push_back(&observation);
      }
    }
    std::stable_sort(
      usable.begin(), usable.end(), [this](const Armor* lhs, const Armor* rhs) {
        return L6Telemetry::squaredDistance(lhs->center, image_center_) <
               L6Telemetry::squaredDistance(rhs->center, image_center_);
      });
    return usable;
  }

  [[nodiscard]] bool initializeTarget(
    const std::vector<const Armor*>& armors,
    TimePoint timestamp,
    const Eigen::Isometry3d& T_world_camera)
  {
#ifdef NEWVISION_USE_GTSAM
    if (armors.empty()) {
      return false;
    }
    target_.emplace(
      target_config_, gtsam_config_, armor_config_, calibration_);
    target_->initialize(*armors.front(), timestamp, T_world_camera);
    return true;
#else
    static_cast<void>(armors);
    static_cast<void>(timestamp);
    static_cast<void>(T_world_camera);
    return false;
#endif
  }

  [[nodiscard]] bool updateTarget(
    const std::vector<const Armor*>& armors,
    TimePoint timestamp,
    const Eigen::Isometry3d& T_world_camera)
  {
#ifdef NEWVISION_USE_GTSAM
    return target_ && target_->update(armors, timestamp, T_world_camera);
#else
    static_cast<void>(armors);
    static_cast<void>(timestamp);
    static_cast<void>(T_world_camera);
    return false;
#endif
  }

  void resetTracking() noexcept
  {
    state_ = TrackState::Lost;
#ifdef NEWVISION_USE_GTSAM
    target_.reset();
#endif
  }

  L1Sensor::CameraCalibration calibration_;
  ArmorConfig armor_config_;
  TargetConfig target_config_;
  Config gtsam_config_;
  PnpSolver pnp_solver_;
  cv::Point2f image_center_{};
  bool ready_{false};
  TrackState state_{TrackState::Lost};
#ifdef NEWVISION_USE_GTSAM
  std::optional<Target> target_;
#endif
  std::optional<TimePoint> last_timestamp_;
  std::vector<Armor> observations_;
};

Tracker::Tracker(
  const L1Sensor::CameraCalibration& calibration,
  ArmorConfig armor_config,
  TargetConfig target_config,
  Config gtsam_config)
{
  if (!available()) {
    throw std::runtime_error(
      "estimator backend 'gtsam' was requested but GTSAM is not compiled in; "
      "rebuild with: xmake f --use_gtsam=y");
  }
  impl_ = std::make_unique<Impl>(
    calibration, armor_config, target_config, gtsam_config);
}

Tracker::~Tracker() = default;

bool Tracker::ready() const noexcept { return impl_ && impl_->ready(); }
TrackState Tracker::state() const noexcept
{
  return impl_ ? impl_->state() : TrackState::Lost;
}
EstimatorBackend Tracker::backend() const noexcept { return EstimatorBackend::Gtsam; }

std::optional<TrackedTarget> Tracker::track(
  const std::vector<L2Perception::Armor>& detections,
  const std::optional<Eigen::Quaterniond>& q_world_barrel,
  TimePoint timestamp)
{
  return impl_ ? impl_->track(detections, q_world_barrel, timestamp) : std::nullopt;
}

const std::vector<Armor>& Tracker::observations() const noexcept
{
  static const std::vector<Armor> empty;
  return impl_ ? impl_->observations() : empty;
}

std::vector<Eigen::Vector4d> Tracker::targetArmorPoses() const
{
  return impl_ ? impl_->targetArmorPoses() : std::vector<Eigen::Vector4d>{};
}

void Tracker::reset() noexcept
{
  if (impl_) {
    impl_->reset();
  }
}

}  // namespace L3Estimation::GtsamEst
