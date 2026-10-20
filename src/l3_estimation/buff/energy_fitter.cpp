#include "l3_estimation/buff/energy_fitter.hpp"

#include <Eigen/Dense>

#include <cmath>

namespace L3Estimation::Buff {
namespace {

double sampleWeight(double t, double reference)
{
  return std::exp2((t - reference) / EnergyFitter::kWeightHalfLifeSeconds);
}

}  // namespace

void EnergyFitter::push(double t, double theta)
{
  buffer_.push_back({t, theta});
  while (!buffer_.empty() && buffer_.back().t - buffer_.front().t > kWindowSeconds) {
    buffer_.pop_front();
  }
}

void EnergyFitter::reset() noexcept
{
  buffer_.clear();
}

template<typename Prediction>
double weightedCostImpl(
  const std::deque<EnergyFitter::Point>& buffer, Prediction&& prediction)
{
  if (buffer.empty()) return std::numeric_limits<double>::max();
  const double reference = buffer.back().t;
  double weighted_sum = 0.0;
  double weight_sum = 0.0;
  for (const auto& point : buffer) {
    const double weight = sampleWeight(point.t, reference);
    const double difference = point.theta - prediction(point.t);
    weighted_sum += weight * difference * difference;
    weight_sum += weight;
  }
  return weighted_sum / weight_sum;
}

auto EnergyFitter::fitLinear() const -> std::optional<LinearResult>
{
  if (buffer_.size() < 2 ||
      buffer_.back().t - buffer_.front().t < kMinFitSeconds) return std::nullopt;
  const double reference = buffer_.back().t;
  double sum_weight = 0.0;
  double sum_t = 0.0;
  double sum_theta = 0.0;
  double sum_tt = 0.0;
  double sum_t_theta = 0.0;
  for (const auto& point : buffer_) {
    const double weight = sampleWeight(point.t, reference);
    sum_weight += weight;
    sum_t += weight * point.t;
    sum_theta += weight * point.theta;
    sum_tt += weight * point.t * point.t;
    sum_t_theta += weight * point.t * point.theta;
  }
  const double denominator = sum_weight * sum_tt - sum_t * sum_t;
  if (std::abs(denominator) < 1e-12) return std::nullopt;
  const double speed = (sum_weight * sum_t_theta - sum_t * sum_theta) /
                       denominator;
  const double C = (sum_theta - speed * sum_t) / sum_weight;
  const auto prediction = [C, speed](double t) { return C + speed * t; };
  return LinearResult{C, speed, weightedCostImpl(buffer_, prediction)};
}

auto EnergyFitter::fitSine() const -> std::optional<FitResult>
{
  if (buffer_.size() < 2 ||
      buffer_.back().t - buffer_.front().t < kMinFitSeconds) return std::nullopt;

  const auto count = static_cast<Eigen::Index>(buffer_.size());
  Eigen::MatrixXd X(count, 4);
  Eigen::VectorXd y(count);
  Eigen::VectorXd sqrt_weights(count);
  const double reference = buffer_.back().t;
  double weight_sum = 0.0;
  for (Eigen::Index i = 0; i < count; ++i) {
    const double t = buffer_[static_cast<std::size_t>(i)].t;
    const double weight = sampleWeight(t, reference);
    y(i) = buffer_[static_cast<std::size_t>(i)].theta;
    X(i, 0) = 1.0;
    X(i, 1) = t;
    sqrt_weights(i) = std::sqrt(weight);
    weight_sum += weight;
  }
  const auto weighted_y = y.cwiseProduct(sqrt_weights);

  constexpr int kSweepSteps = 41;
  constexpr double kOmegaMin = 1.80;
  constexpr double kOmegaMax = 2.20;
  double best_mse = std::numeric_limits<double>::max();
  double best_C = 0.0;
  double best_v = 0.0;
  double best_A = 0.0;
  double best_B = 0.0;
  double best_omega = 0.0;

  for (int step = 0; step <= kSweepSteps; ++step) {
    const double omega = kOmegaMin + step * (kOmegaMax - kOmegaMin) / kSweepSteps;
    for (Eigen::Index i = 0; i < count; ++i) {
      const double t = buffer_[static_cast<std::size_t>(i)].t;
      X(i, 2) = std::cos(omega * t);
      X(i, 3) = std::sin(omega * t);
    }
    Eigen::MatrixXd weighted_X = X;
    for (Eigen::Index i = 0; i < count; ++i) weighted_X.row(i) *= sqrt_weights(i);
    const Eigen::Vector4d coefficients =
      weighted_X.colPivHouseholderQr().solve(weighted_y);
    const double mse = (weighted_y - weighted_X * coefficients).squaredNorm() /
                       weight_sum;
    if (mse < best_mse) {
      best_mse = mse;
      best_C = coefficients(0);
      best_v = coefficients(1);
      best_A = coefficients(2);
      best_B = coefficients(3);
      best_omega = omega;
    }
  }

  const double a = best_omega * std::sqrt(best_A * best_A + best_B * best_B);
  const double phi = std::atan2(best_B, -best_A);
  const auto prediction = [=](double t) {
    return best_C + best_v * t - a / best_omega * std::cos(best_omega * t + phi);
  };
  return FitResult{best_C, best_v, a, best_omega, phi,
                   weightedCostImpl(buffer_, prediction)};
}

}  // namespace L3Estimation::Buff
