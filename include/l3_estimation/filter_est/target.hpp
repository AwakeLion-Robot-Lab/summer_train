#pragma once

#include "l3_estimation/filter_est/config.hpp"
#include "l3_estimation/filter_est/ekf.hpp"
#include "l3_estimation/tracked_target.hpp"
#include "l3_estimation/types.hpp"

#include <vector>

namespace L3Estimation::FilterEst {

// EKF 后端自己的整车目标。滤波器、NIS 窗口、关联计数都封闭在 FilterEst 内，
// 对外只通过 snapshot() 产生后端中立的 TrackedTarget。
class Target
{
public:
  Target(
    const Armor& armor,
    TimePoint timestamp,
    double radius,
    int armor_count,
    const Eigen::VectorXd& initial_covariance_diagonal,
    L3Estimation::TargetConfig target_config = {},
    TargetConfig filter_config = {});

  void predict(TimePoint timestamp);
  void update(const Armor& armor);

  [[nodiscard]] ArmorName name() const noexcept { return name_; }
  [[nodiscard]] ArmorType armorType() const noexcept { return armor_type_; }
  [[nodiscard]] std::vector<Eigen::Vector4d> armorPoses() const;
  [[nodiscard]] TrackedTarget snapshot() const;
  [[nodiscard]] bool diverged() const;
  [[nodiscard]] bool converged();
  [[nodiscard]] const ExtendedKalmanFilter& filter() const noexcept { return filter_; }

private:
  void updateObservation(const Armor& armor, int armor_id);
  [[nodiscard]] Eigen::MatrixXd observationJacobian(int armor_id) const;

  L3Estimation::TargetConfig target_config_{};
  TargetConfig filter_config_{};
  ArmorName name_{ArmorName::Unknown};
  ArmorType armor_type_{ArmorType::Small};
  int armor_count_{4};
  int update_count_{0};
  bool converged_{false};
  bool jumped_{false};
  int last_id_{0};
  ExtendedKalmanFilter filter_;
  TimePoint timestamp_{};
};

}  // namespace L3Estimation::FilterEst
