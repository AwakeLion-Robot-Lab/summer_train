#include "l3_estimation/armor/light_residual.hpp"

#include <Eigen/Cholesky>

#include <array>
#include <cmath>
#include <cstddef>

namespace L3Estimation {

std::optional<double> chi2(
  const Eigen::VectorXd & residual, const Eigen::MatrixXd & covariance)
{
  if (
    residual.size() <= 0 || covariance.rows() != residual.size() ||
    covariance.cols() != residual.size()) {
    return std::nullopt;
  }
  const Eigen::LLT<Eigen::MatrixXd> llt(covariance);
  if (llt.info() != Eigen::Success) {
    return std::nullopt;
  }
  return residual.dot(llt.solve(residual));
}

LightResidual analyzeLight(
  const Eigen::VectorXd & innovation, const std::vector<LightAxis> & lights,
  bool has_depth_diff)
{
  LightResidual result;

  const int light_count = static_cast<int>(lights.size());
  if (light_count <= 0 || innovation.size() < light_count * kLightMeasureSize) {
    return result;
  }

  double along_sq = 0.0;
  double perp_sq = 0.0;
  // 下标与 LightResidual 的四个通道同序：横移⊥、沿移∥、倾角、长度。
  std::array<double, 4> sum{};
  std::array<double, 4> sum_sq{};

  for (int i = 0; i < light_count; ++i) {
    const int base = i * kLightMeasureSize;
    const Eigen::Vector2d & e = lights[i].direction;
    const Eigen::Vector2d n(-e.y(), e.x());
    const Eigen::Vector2d r_top = innovation.segment<2>(base + endpoint::TOP_U);
    const Eigen::Vector2d r_bottom = innovation.segment<2>(base + endpoint::BOTTOM_U);

    const double top_along = r_top.dot(e);
    const double bottom_along = r_bottom.dot(e);
    const double top_perp = r_top.dot(n);
    const double bottom_perp = r_bottom.dot(n);

    along_sq += top_along * top_along + bottom_along * bottom_along;
    perp_sq += top_perp * top_perp + bottom_perp * bottom_perp;

    // 长度为 0 的灯条不该出现在这里，真出现了就把倾角记 0，别除出 inf。
    const double length = lights[i].length;
    const std::array<double, 4> channel{
      (top_perp + bottom_perp) / 2.0,
      (top_along + bottom_along) / 2.0,
      length > 1e-6 ? (top_perp - bottom_perp) / length : 0.0,
      bottom_along - top_along};
    for (std::size_t k = 0; k < channel.size(); ++k) {
      sum[k] += channel[k];
      sum_sq[k] += channel[k] * channel[k];
    }
  }

  const double samples = static_cast<double>(light_count);
  const std::array<LightResidual::Channel *, 4> out{
    &result.shift_perp, &result.shift_along, &result.tilt, &result.length};
  for (std::size_t k = 0; k < out.size(); ++k) {
    out[k]->mean = sum[k] / samples;
    out[k]->rms = std::sqrt(sum_sq[k] / samples);
  }

  result.along_rms_px = std::sqrt(along_sq / (2.0 * samples));
  result.perp_rms_px = std::sqrt(perp_sq / (2.0 * samples));
  result.light_count = light_count;

  const int depth_index = light_count * kLightMeasureSize;
  if (has_depth_diff && depth_index < innovation.size()) {
    result.depth_diff_m = innovation[depth_index];
  }

  return result;
}

}  // namespace L3Estimation
