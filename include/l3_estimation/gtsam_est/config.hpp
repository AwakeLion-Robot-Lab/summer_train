#pragma once

#include <chrono>
#include <numbers>

namespace L3Estimation::GtsamEst {

// GTSAM 后端独有的因子噪声和关联参数。所有噪声都是标准差 sigma。
struct Config {
  double max_match_distance{1.5};
  double max_match_yaw_diff{60.0 * std::numbers::pi / 180.0};

  int first_update_batch_size{1};
  std::chrono::milliseconds lost_threshold{300};

  double translation_prior_sigma{0.1};
  double velocity_prior_sigma{0.5};
  double yaw_prior_sigma{9.0};
  double vyaw_prior_sigma{9.0};
  double radius_prior_sigma{1e-4};
  double dz_prior_sigma{0.05};
  double default_radius{0.26};
  double radius_min{0.15};
  double radius_max{0.5};
  double default_dz{0.0};

  double translation_factor_sigma{0.01};
  double velocity_factor_sigma{0.03};
  double yaw_factor_sigma{0.01};
  double vyaw_factor_sigma{0.2};

  double obs_tangential_sigma{0.005};
  double obs_radial_sigma{0.5};
  double obs_height_sigma{0.01};
  double obs_yaw_sigma{0.001};
  double obs_pixel_sigma{0.1};
};

}  // namespace L3Estimation::GtsamEst
