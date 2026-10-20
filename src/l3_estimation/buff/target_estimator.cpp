#include "l3_estimation/buff/target_estimator.hpp"

#include "l3_estimation/buff/energy_fitter.hpp"
#include "l3_estimation/buff/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

namespace L3Estimation::Buff {
namespace {

constexpr int kX = 0;
constexpr int kY = 1;
constexpr int kZ = 2;
constexpr int kW = 3;
constexpr int kA = 4;
constexpr int kPsi = 5;
constexpr int kIcon = 0;
constexpr int kFeatureCount = 6;

Eigen::Matrix3d rosToOpenCv()
{
  Eigen::Matrix3d matrix;
  matrix << 0.0, -1.0, 0.0,
            0.0, 0.0, -1.0,
            1.0, 0.0, 0.0;
  return matrix;
}

double normalizeAngle(double angle)
{
  return std::remainder(angle, 2.0 * std::numbers::pi);
}

std::vector<std::optional<int>> hungarianAssign(
  const Eigen::MatrixXd& cost, double gate)
{
  const int rows = static_cast<int>(cost.rows());
  const int cols = static_cast<int>(cost.cols());
  if (rows == 0) return {};
  if (cols == 0) return std::vector<std::optional<int>>(rows, std::nullopt);
  const int n = rows + cols;
  const double huge = std::numeric_limits<double>::max() / 4.0;
  Eigen::MatrixXd augmented = Eigen::MatrixXd::Constant(n, n, huge);
  augmented.topLeftCorner(rows, cols) = cost;
  for (int i = 0; i < rows; ++i) {
    for (int j = cols; j < n; ++j) {
      augmented(i, j) = j - cols == i ? gate : huge;
    }
  }
  for (int i = rows; i < n; ++i) {
    for (int j = 0; j < cols; ++j) {
      augmented(i, j) = i - rows == j ? 0.0 : huge;
    }
    for (int j = cols; j < n; ++j) augmented(i, j) = 0.0;
  }

  std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minimum(n + 1, std::numeric_limits<double>::max());
    std::vector<char> used(n + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = std::numeric_limits<double>::max();
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) continue;
        const double current = augmented(i0 - 1, j - 1) - u[i0] - v[j];
        if (current < minimum[j]) {
          minimum[j] = current;
          way[j] = j0;
        }
        if (minimum[j] < delta) {
          delta = minimum[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minimum[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<std::optional<int>> result(rows, std::nullopt);
  for (int j = 1; j <= n; ++j) {
    if (p[j] != 0 && p[j] - 1 < rows && j - 1 < cols) {
      result[p[j] - 1] = j - 1;
    }
  }
  return result;
}

// 与源 SingleRunePnpSolution 使用同一套 Icon/T/L/B/R 排序。PnP 结果的
// 重投影误差必须在五个真实观测点上计算，不能只拿 seed 圆心重复代替四个臂端点。
std::optional<std::array<cv::Point2f, 5>> orderedImagePoints(
  const L2Perception::Buff::Icon& icon,
  const L2Perception::Buff::Bullseye& bullseye)
{
  const cv::Point2f icon_point = icon.center;
  const cv::Point2f center = bullseye.center;
  std::size_t bottom = 0;
  std::size_t top = 0;
  double min_distance = std::numeric_limits<double>::max();
  double max_distance = -std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < bullseye.corners.size(); ++i) {
    const double distance = cv::norm(bullseye.corners[i] - icon_point);
    if (distance < min_distance) { min_distance = distance; bottom = i; }
    if (distance > max_distance) { max_distance = distance; top = i; }
  }
  if (bottom == top) return std::nullopt;
  std::array<std::size_t, 2> remaining{};
  std::size_t count = 0;
  for (std::size_t i = 0; i < bullseye.corners.size(); ++i) {
    if (i == bottom || i == top) continue;
    if (count >= remaining.size()) return std::nullopt;
    remaining[count++] = i;
  }
  if (count != remaining.size()) return std::nullopt;
  const cv::Point2f z_raw = bullseye.corners[top] - bullseye.corners[bottom];
  const double z_length = cv::norm(z_raw);
  if (z_length <= 1e-6) return std::nullopt;
  const cv::Point2d z_axis{z_raw.x / z_length, z_raw.y / z_length};
  const cv::Point2d y_axis{z_axis.y, -z_axis.x};
  const auto projection = [&](std::size_t i) {
    return (bullseye.corners[i].x - center.x) * y_axis.x +
           (bullseye.corners[i].y - center.y) * y_axis.y;
  };
  const auto left = projection(remaining[0]) > projection(remaining[1])
    ? bullseye.corners[remaining[0]] : bullseye.corners[remaining[1]];
  const auto right = projection(remaining[0]) > projection(remaining[1])
    ? bullseye.corners[remaining[1]] : bullseye.corners[remaining[0]];
  return std::array<cv::Point2f, 5>{icon_point, bullseye.corners[top], left,
                                    bullseye.corners[bottom], right};
}

std::array<cv::Point3f, 5> pagePointsOpenCv()
{
  const Eigen::Matrix3d ros_to_cv = rosToOpenCv();
  const std::array<Eigen::Vector3d, 5> points{
    PagePoints::icon(), PagePoints::top(), PagePoints::left(),
    PagePoints::bottom(), PagePoints::right()};
  std::array<cv::Point3f, 5> result{};
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d p = ros_to_cv * points[i];
    result[i] = cv::Point3f{static_cast<float>(p.x()), static_cast<float>(p.y()),
                            static_cast<float>(p.z())};
  }
  return result;
}

}  // namespace

void State::transition(double seconds) noexcept
{
  if (sine_valid) {
    const double old_phase = sine_phase;
    sine_t += seconds;
    sine_phase += sine_omega * seconds;
    if (std::abs(sine_omega) > 1e-12) {
      rotation_angle += sine_v * seconds + sine_a / sine_omega *
                        (std::cos(old_phase) - std::cos(sine_phase));
    } else {
      rotation_angle += rotation_speed * seconds;
    }
    rotation_speed = sine_v + sine_a * std::sin(sine_phase);
  } else {
    rotation_angle += rotation_speed * seconds;
  }
}

std::vector<Eigen::Vector3d> State::aimpoints() const
{
  const auto duration = std::chrono::seconds{sine_valid ? 6 : 3};
  if (std::chrono::steady_clock::now() - start_timestamp < duration) return {};

  const Eigen::Matrix3d face =
    Eigen::AngleAxisd{face_yaw, Eigen::Vector3d::UnitZ()}.toRotationMatrix();
  constexpr std::array<double, 5> blade_degrees{0.0, 72.0, 144.0, 216.0, 288.0};
  std::vector<Eigen::Vector3d> result;
  for (std::size_t i = 0; i < blade_degrees.size(); ++i) {
    if (!inactive[i]) continue;
    const double alpha = rotation_angle + blade_degrees[i] * std::numbers::pi / 180.0;
    const Eigen::Vector3d local{0.0, -kGlobalRadius * std::sin(alpha),
                                kGlobalRadius * std::cos(alpha)};
    result.push_back(Eigen::Vector3d{x, y, z} + face * local);
    break;
  }
  return result;
}

struct TargetEstimator::Impl {
  using StateVector = Eigen::Matrix<double, 6, 1>;
  using Covariance = Eigen::Matrix<double, 6, 6>;
  using ObservationNoise = Eigen::Matrix<double, 2, 2>;
  using Jacobian = Eigen::Matrix<double, 2, 6>;

  struct Observable {
    std::array<Eigen::Vector2d, kFeatureCount> projected{};
    std::array<Eigen::Vector3d, kFeatureCount> in_camera{};
    std::array<bool, kFeatureCount> visible{};
  } observable;

  Config config;
  L1Sensor::CameraCalibration calibration;
  Eigen::Quaterniond q_world_camera{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d t_world_camera{Eigen::Vector3d::Zero()};
  StateVector posterior{StateVector::Zero()};
  Covariance covariance{Covariance::Identity()};
  std::array<bool, 5> blade_inactive{};
  std::array<TimePoint, 5> blade_timeout{};
  Addition addition{};
  TimePoint init_timestamp{};
  TimePoint current_stamp{};
  std::size_t update_count{0};
  EnergyFitter fitter;
  TimePoint force_sine_until{};

  Impl(Config value, const L1Sensor::CameraCalibration& camera)
  : config(value), calibration(camera)
  {
    auto diagonal = StateVector{};
    diagonal << 64.0, 64.0, 64.0, 100.0, 25.0, 10.0;
    covariance = diagonal.asDiagonal();
  }

  Eigen::Vector2d projectPoint(const Eigen::Vector3d& world) const
  {
    const Eigen::Vector3d camera = q_world_camera.conjugate() *
                                   (world - t_world_camera);
    const Eigen::Vector3d point = rosToOpenCv() * camera;
    std::vector<cv::Point3f> object{{static_cast<float>(point.x()),
                                     static_cast<float>(point.y()),
                                     static_cast<float>(point.z())}};
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0),
                      calibration.camera_matrix,
                      calibration.distortion_coefficients, projected);
    if (projected.empty()) return Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN());
    return {projected[0].x, projected[0].y};
  }

  void updateObservable(const StateVector& state)
  {
    const Eigen::Matrix3d face =
      Eigen::AngleAxisd{state[kPsi], Eigen::Vector3d::UnitZ()}.toRotationMatrix();
    const Eigen::Vector3d center{state[kX], state[kY], state[kZ]};
    constexpr std::array<double, 5> blade_degrees{0.0, 72.0, 144.0, 216.0, 288.0};
    for (int i = 0; i < kFeatureCount; ++i) {
      Eigen::Vector3d local;
      if (i == kIcon) {
        local = PagePoints::icon();
      } else {
        const double alpha = state[kA] + blade_degrees[static_cast<std::size_t>(i - 1)] *
                             std::numbers::pi / 180.0;
        local = {0.0, -kGlobalRadius * std::sin(alpha),
                 kGlobalRadius * std::cos(alpha)};
      }
      const Eigen::Vector3d world = center + face * local;
      const Eigen::Vector3d camera = q_world_camera.conjugate() *
                                     (world - t_world_camera);
      observable.in_camera[i] = camera;
      const Eigen::Vector3d cv_point = rosToOpenCv() * camera;
      if (cv_point.z() > 0.1 && cv_point.allFinite()) {
        observable.projected[i] = projectPoint(world);
        observable.visible[i] = observable.projected[i].allFinite();
      } else {
        observable.projected[i].setZero();
        observable.visible[i] = false;
      }
    }
  }

  Jacobian makeJacobian(int feature_id) const
  {
    Jacobian result = Jacobian::Zero();
    const auto& camera = observable.in_camera[feature_id];
    const Eigen::Vector3d point = rosToOpenCv() * camera;
    if (point.z() < 0.1) return result;
    const double fx = calibration.camera_matrix.at<double>(0, 0);
    const double fy = calibration.camera_matrix.at<double>(1, 1);
    Eigen::Matrix<double, 2, 3> projection;
    projection << fx / point.z(), 0.0, -fx * point.x() / (point.z() * point.z()),
                  0.0, fy / point.z(), -fy * point.y() / (point.z() * point.z());
    const Eigen::Matrix3d cam_ros_jac = rosToOpenCv();
    const Eigen::Matrix3d r_camera_world = q_world_camera.conjugate().toRotationMatrix();
    Eigen::Matrix<double, 3, 6> position = Eigen::Matrix<double, 3, 6>::Zero();
    position(0, kX) = 1.0;
    position(1, kY) = 1.0;
    position(2, kZ) = 1.0;
    const Eigen::Matrix3d face =
      Eigen::AngleAxisd{posterior[kPsi], Eigen::Vector3d::UnitZ()}.toRotationMatrix();
    Eigen::Matrix3d dface;
    dface << -std::sin(posterior[kPsi]), -std::cos(posterior[kPsi]), 0.0,
              std::cos(posterior[kPsi]), -std::sin(posterior[kPsi]), 0.0,
              0.0, 0.0, 0.0;
    Eigen::Vector3d local = Eigen::Vector3d::Zero();
    if (feature_id == kIcon) {
      local = PagePoints::icon();
    } else {
      constexpr std::array<double, 5> degrees{0.0, 72.0, 144.0, 216.0, 288.0};
      const double alpha = posterior[kA] + degrees[static_cast<std::size_t>(feature_id - 1)] *
                           std::numbers::pi / 180.0;
      local = {0.0, -kGlobalRadius * std::sin(alpha),
               kGlobalRadius * std::cos(alpha)};
      Eigen::Vector3d dlocal{
        0.0, -kGlobalRadius * std::cos(alpha), -kGlobalRadius * std::sin(alpha)};
      position.col(kA) = face * dlocal;
    }
    position.col(kPsi) = dface * local;
    result.noalias() = projection * cam_ros_jac * r_camera_world * position;
    return result;
  }

  bool updateOne(const Eigen::Vector2d& observed, int feature_id) noexcept
  {
    const Eigen::Vector2d innovation = observed - observable.projected[feature_id];
    if (!innovation.allFinite()) return false;
    const Jacobian jacobian = makeJacobian(feature_id);
    const ObservationNoise noise =
      ObservationNoise::Identity() * config.noise_observation;
    const ObservationNoise innovation_cov =
      jacobian * covariance * jacobian.transpose() + noise;
    const auto inverse = innovation_cov.inverse();
    if (!inverse.allFinite()) return false;
    const Eigen::Matrix<double, 6, 2> gain =
      covariance * jacobian.transpose() * inverse;
    StateVector next = posterior + gain * innovation;
    if (!next.allFinite()) return false;
    next[kPsi] = normalizeAngle(next[kPsi]);
    const Covariance complement = Covariance::Identity() - gain * jacobian;
    Covariance next_cov = complement * covariance * complement.transpose();
    next_cov += (gain * noise * gain.transpose()).eval();
    next_cov = 0.5 * (next_cov + next_cov.transpose());
    posterior = next;
    covariance = next_cov;
    updateObservable(posterior);
    return true;
  }

  void resetCovariance()
  {
    auto diagonal = StateVector{};
    diagonal << 64.0, 64.0, 64.0, 100.0, 25.0, 10.0;
    covariance = diagonal.asDiagonal();
  }

  State state() const noexcept
  {
    State result;
    result.x = posterior[kX];
    result.y = posterior[kY];
    result.z = posterior[kZ];
    result.start_timestamp = init_timestamp;
    result.timestamp = current_stamp;
    result.rotation_speed = sine_valid()
      ? sine_v + sine_a * std::sin(sine_phase)
      : (use_prediction_speed() ? prediction_speed() : posterior[kW]);
    result.rotation_angle = posterior[kA];
    result.face_yaw = posterior[kPsi];
    result.inactive = blade_inactive;
    result.use_prediction_speed = use_prediction_speed();
    result.prediction_cost = sine_valid() ? sine_cost() : prediction_cost();
    result.sine_C = sine_C;
    result.sine_v = sine_v;
    result.sine_a = sine_a;
    result.sine_omega = sine_omega;
    result.sine_phase = sine_phase;
    result.sine_t = sine_t;
    result.sine_valid = sine_valid();
    result.update_count = update_count;
    return result;
  }

  double prediction_speed() const noexcept { return fit_speed; }
  bool use_prediction_speed() const noexcept { return fit_linear_valid; }
  double prediction_cost() const noexcept { return fit_linear_cost; }
  bool sine_valid() const noexcept { return fit_sine_valid; }
  double sine_cost() const noexcept { return fit_sine_cost; }
  double sine_C{0.0};
  double sine_v{0.0};
  double sine_a{0.0};
  double sine_omega{0.0};
  double sine_phase{0.0};
  double sine_t{0.0};
  bool fit_sine_valid{false};
  bool fit_linear_valid{false};
  double fit_speed{0.0};
  double fit_linear_cost{0.0};
  double fit_sine_cost{0.0};
};

TargetEstimator::TargetEstimator(
  Config config, const L1Sensor::CameraCalibration& calibration)
: impl_(std::make_unique<Impl>(config, calibration))
{
}

TargetEstimator::~TargetEstimator() = default;

void TargetEstimator::updateTransform(
  const Eigen::Quaterniond& q_world_camera, const Eigen::Vector3d& t_world_camera)
{
  impl_->q_world_camera = q_world_camera;
  impl_->t_world_camera = t_world_camera;
  impl_->updateObservable(impl_->posterior);
}

bool TargetEstimator::init(
  std::span<const L2Perception::Buff::Icon> icons,
  std::span<const L2Perception::Buff::Bullseye> bullseyes,
  TimePoint timestamp)
{
  if (icons.empty()) return false;
  std::vector<const L2Perception::Buff::Bullseye*> inactive;
  for (const auto& bullseye : bullseyes) {
    if (!bullseye.active) inactive.push_back(&bullseye);
  }
  if (inactive.empty() || inactive.size() > 2) return false;

  struct Candidate {
    Eigen::Vector3d center;
    double face_yaw;
    double angle;
    cv::Point2f icon;
    cv::Point2f seed;
    int inliers;
    double icon_error;
    double seed_error;
    double seed_sse;
    double seed_max;
    double inactive_sse;
  };
  std::optional<Candidate> best;
  PnpSolver pnp(impl_->calibration);
  for (const auto& icon : icons) {
    for (const auto* seed : inactive) {
      const auto solution = pnp.solve(icon, *seed);
      if (!solution) continue;
      const auto image_points = orderedImagePoints(icon, *seed);
      if (!image_points) continue;
      std::vector<cv::Point3f> camera_points;
      const auto local_points = pagePointsOpenCv();
      camera_points.reserve(local_points.size());
      for (const auto& local : local_points) {
        const Eigen::Vector3d ros_local = rosToOpenCv().transpose() * Eigen::Vector3d{
          local.x, local.y, local.z};
        const Eigen::Vector3d ros_camera = solution->orientation_camera * ros_local +
                                           solution->translation_camera;
        const Eigen::Vector3d cv_camera = rosToOpenCv() * ros_camera;
        camera_points.emplace_back(static_cast<float>(cv_camera.x()),
                                   static_cast<float>(cv_camera.y()),
                                   static_cast<float>(cv_camera.z()));
      }
      std::vector<cv::Point2f> projected;
      try {
        cv::projectPoints(camera_points, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0),
                          impl_->calibration.camera_matrix,
                          impl_->calibration.distortion_coefficients, projected);
      } catch (const cv::Exception&) {
        continue;
      }
      if (projected.size() != image_points->size()) continue;
      double seed_sse = 0.0;
      double seed_max = 0.0;
      for (std::size_t i = 0; i < projected.size(); ++i) {
        const double error = cv::norm(projected[i] - (*image_points)[i]);
        seed_sse += error * error;
        seed_max = std::max(seed_max, error);
      }
      const double seed_mean = std::sqrt(seed_sse / 5.0);
      if (seed_mean > impl_->config.init_seed_mean_error ||
          seed_max > impl_->config.init_seed_max_error) continue;
      const Eigen::Quaterniond q_page =
        (impl_->q_world_camera * solution->orientation_camera).normalized();
      const Eigen::Vector3d center =
        impl_->q_world_camera * solution->translation_camera + impl_->t_world_camera;
      const Eigen::Vector3d face_x = q_page * Eigen::Vector3d::UnitX();
      if (face_x.head<2>().squaredNorm() <= 1e-6) continue;
      const double pitch = std::asin(std::abs(face_x.z()));
      if (pitch > impl_->config.init_pitch_bound_deg * std::numbers::pi / 180.0) continue;
      const double face_yaw = std::atan2(face_x.y(), face_x.x());
      const Eigen::Vector3d camera_face_x =
        impl_->q_world_camera.conjugate() *
        Eigen::Vector3d{std::cos(face_yaw), std::sin(face_yaw), 0.0};
      if (camera_face_x.x() <= 0.0) continue;
      const Eigen::Matrix3d inverse_yaw =
        Eigen::AngleAxisd{-face_yaw, Eigen::Vector3d::UnitZ()}.toRotationMatrix();
      const Eigen::Matrix3d seed_local = inverse_yaw * q_page.toRotationMatrix();
      const double seed_angle = std::atan2(-seed_local(1, 2), seed_local(2, 2));

      Impl::StateVector candidate_state = Impl::StateVector::Zero();
      candidate_state[kX] = center.x();
      candidate_state[kY] = center.y();
      candidate_state[kZ] = center.z();
      candidate_state[kA] = seed_angle;
      candidate_state[kPsi] = face_yaw;
      const auto previous = impl_->posterior;
      impl_->posterior = candidate_state;
      impl_->updateObservable(candidate_state);
      if (!impl_->observable.visible[kIcon] || !impl_->observable.visible[1]) {
        impl_->posterior = previous;
        continue;
      }
      const Eigen::Vector2d icon_observation{icon.center.x, icon.center.y};
      const Eigen::Vector2d seed_observation{seed->center.x, seed->center.y};
      const double icon_error =
        (icon_observation - impl_->observable.projected[kIcon]).squaredNorm();
      const double seed_error =
        (seed_observation - impl_->observable.projected[1]).squaredNorm();
      if (seed_error > impl_->config.init_center_gate * impl_->config.init_center_gate) {
        impl_->posterior = previous;
        continue;
      }
      int inliers = 0;
      double inactive_sse = std::numeric_limits<double>::max();
      if (inactive.size() > 1) {
        inactive_sse = 0.0;
        for (std::size_t index = 1; index < inactive.size(); ++index) {
          const Eigen::Vector2d observation{
            inactive[index]->center.x, inactive[index]->center.y};
          double best_error = std::numeric_limits<double>::max();
          for (int feature = 2; feature < kFeatureCount; ++feature) {
            if (!impl_->observable.visible[feature]) continue;
            best_error = std::min(best_error,
              (observation - impl_->observable.projected[feature]).squaredNorm());
          }
          if (best_error <= impl_->config.init_center_gate * impl_->config.init_center_gate) {
            ++inliers;
            inactive_sse += best_error;
          }
        }
      } else {
        inactive_sse = 0.0;
      }
      Candidate candidate{center, face_yaw, seed_angle, icon.center, seed->center,
                          inliers, icon_error, seed_error, seed_sse, seed_max,
                          inactive_sse};
      const auto better = [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.inliers != rhs.inliers) return lhs.inliers > rhs.inliers;
        if (lhs.seed_error != rhs.seed_error) return lhs.seed_error < rhs.seed_error;
        if (lhs.icon_error != rhs.icon_error) return lhs.icon_error < rhs.icon_error;
        if (lhs.seed_max != rhs.seed_max) return lhs.seed_max < rhs.seed_max;
        if (lhs.seed_sse != rhs.seed_sse) return lhs.seed_sse < rhs.seed_sse;
        return lhs.inactive_sse < rhs.inactive_sse;
      };
      if (!best || better(candidate, *best)) best = candidate;
      impl_->posterior = previous;
    }
  }
  if (!best) return false;

  impl_->posterior.setZero();
  impl_->posterior[kX] = best->center.x();
  impl_->posterior[kY] = best->center.y();
  impl_->posterior[kZ] = best->center.z();
  impl_->posterior[kA] = best->angle;
  impl_->posterior[kPsi] = best->face_yaw;
  impl_->resetCovariance();
  impl_->init_timestamp = timestamp;
  impl_->current_stamp = timestamp;
  impl_->update_count = 0;
  impl_->fitter.reset();
  impl_->blade_inactive.fill(false);
  impl_->blade_timeout.fill(TimePoint{});
  impl_->force_sine_until = TimePoint{};
  impl_->updateObservable(impl_->posterior);
  impl_->updateOne({best->icon.x, best->icon.y}, 0);
  impl_->updateOne({best->seed.x, best->seed.y}, 1);
  impl_->addition.tracked.clear();
  impl_->addition.predicted.clear();
  for (int feature = 0; feature < kFeatureCount; ++feature) {
    if (impl_->observable.visible[feature]) {
      impl_->addition.predicted.push_back({
        feature, {static_cast<float>(impl_->observable.projected[feature].x()),
                  static_cast<float>(impl_->observable.projected[feature].y())}});
    }
  }
  return true;
}

void TargetEstimator::predict(double dt, TimePoint timestamp) noexcept
{
  impl_->current_stamp = timestamp;
  impl_->posterior[kA] += impl_->posterior[kW] * dt;
  impl_->posterior[kPsi] = normalizeAngle(impl_->posterior[kPsi]);
  Eigen::Matrix<double, 6, 6> transition =
    Eigen::Matrix<double, 6, 6>::Identity();
  transition(kA, kW) = dt;
  Eigen::Matrix<double, 6, 6> noise = Eigen::Matrix<double, 6, 6>::Zero();
  noise.diagonal() << impl_->config.noise_x, impl_->config.noise_y,
    impl_->config.noise_z, impl_->config.noise_rotation_speed,
    impl_->config.noise_rotation_angle, impl_->config.noise_face_yaw;
  impl_->covariance = transition * impl_->covariance * transition.transpose() + noise;
  if (impl_->fit_sine_valid) {
    impl_->sine_t += dt;
    impl_->sine_phase += impl_->sine_omega * dt;
  }
  impl_->updateObservable(impl_->posterior);
}

bool TargetEstimator::correct(
  std::span<const L2Perception::Buff::Icon> icons,
  std::span<const L2Perception::Buff::Bullseye> bullseyes) noexcept
{
  impl_->updateObservable(impl_->posterior);
  for (std::size_t i = 0; i < impl_->blade_inactive.size(); ++i) {
    if (impl_->current_stamp >= impl_->blade_timeout[i]) {
      impl_->blade_inactive[i] = false;
    }
  }
  impl_->addition.predicted.clear();
  impl_->addition.tracked.clear();
  for (int feature = 0; feature < kFeatureCount; ++feature) {
    if (impl_->observable.visible[feature]) {
      impl_->addition.predicted.push_back({
        feature, {static_cast<float>(impl_->observable.projected[feature].x()),
                  static_cast<float>(impl_->observable.projected[feature].y())}});
    }
  }

  struct Observation { Eigen::Vector2d pixel; bool icon; bool inactive; };
  std::vector<Observation> observations;
  for (const auto& icon : icons) observations.push_back({{icon.center.x, icon.center.y}, true, false});
  for (const auto& bullseye : bullseyes) {
    observations.push_back({{bullseye.center.x, bullseye.center.y}, false, !bullseye.active});
  }
  if (observations.empty()) return false;

  const int count = static_cast<int>(observations.size());
  constexpr double infinite = std::numeric_limits<double>::max() / 4.0;
  Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(count, kFeatureCount, infinite);
  std::array<Impl::Jacobian, kFeatureCount> jacobians{};
  std::array<Impl::ObservationNoise, kFeatureCount> covariances{};
  const Impl::ObservationNoise noise =
    Impl::ObservationNoise::Identity() * impl_->config.noise_observation;
  for (int feature = 0; feature < kFeatureCount; ++feature) {
    jacobians[feature] = impl_->makeJacobian(feature);
    covariances[feature] = jacobians[feature] * impl_->covariance *
      jacobians[feature].transpose() + noise;
  }
  for (int i = 0; i < count; ++i) {
    for (int feature = 0; feature < kFeatureCount; ++feature) {
      if (!impl_->observable.visible[feature]) continue;
      if (observations[i].icon ? feature != kIcon : feature == kIcon) continue;
      const Eigen::Vector2d innovation =
        observations[i].pixel - impl_->observable.projected[feature];
      const Eigen::Vector2d solved = covariances[feature].ldlt().solve(innovation);
      const double distance = innovation.dot(solved);
      if (std::isfinite(distance)) cost(i, feature) = distance;
    }
  }
  const auto assignments = hungarianAssign(cost, impl_->config.gate_threshold);
  int inactive_corrected = 0;
  int active_corrected = 0;
  for (int i = 0; i < count; ++i) {
    if (!assignments[i]) continue;
    const int feature = *assignments[i];
    const Eigen::Vector2d innovation =
      observations[i].pixel - impl_->observable.projected[feature];
    const Eigen::Matrix2d fresh_cov =
      jacobians[feature] * impl_->covariance * jacobians[feature].transpose() + noise;
    const double distance = innovation.dot(fresh_cov.ldlt().solve(innovation));
    if (!std::isfinite(distance) || !impl_->updateOne(observations[i].pixel, feature)) continue;
    if (!observations[i].icon && feature != kIcon) {
      const std::size_t blade = static_cast<std::size_t>(feature - 1);
      if (observations[i].inactive) {
        impl_->blade_inactive[blade] = true;
        impl_->blade_timeout[blade] = impl_->current_stamp +
          std::chrono::duration_cast<TimePoint::duration>(
            std::chrono::duration<double>{impl_->config.inactive_timeout_seconds});
        ++inactive_corrected;
      } else {
        ++active_corrected;
      }
    }
    impl_->addition.tracked.push_back({
      feature, {static_cast<float>(observations[i].pixel.x()),
                static_cast<float>(observations[i].pixel.y())}});
    for (int j = 0; j < kFeatureCount; ++j) {
      jacobians[j] = impl_->makeJacobian(j);
      covariances[j] = jacobians[j] * impl_->covariance *
        jacobians[j].transpose() + noise;
    }
  }

  if (inactive_corrected > 1) {
    impl_->force_sine_until = impl_->current_stamp + std::chrono::seconds{3};
  }
  if (inactive_corrected == 0 && active_corrected == 0) return false;

  ++impl_->update_count;
  const State current = impl_->state();
  const double relative_time = std::chrono::duration<double>(
    impl_->current_stamp - impl_->init_timestamp).count();
  if (relative_time >= 1.0) {
    impl_->fitter.push(relative_time, current.rotation_angle);
    const auto linear = impl_->fitter.fitLinear();
    const auto sine = impl_->fitter.fitSine();
    if (sine && (!linear || impl_->current_stamp < impl_->force_sine_until ||
                 (sine->cost < linear->cost && sine->a >= 0.6))) {
      impl_->fit_linear_valid = false;
      impl_->fit_sine_valid = true;
      impl_->fit_sine_cost = sine->cost;
      impl_->sine_C = sine->C;
      impl_->sine_v = sine->v;
      impl_->sine_a = sine->a;
      impl_->sine_omega = sine->omega;
      impl_->sine_phase = sine->omega * relative_time + sine->phi;
      impl_->sine_t = relative_time;
    } else if (linear) {
      impl_->fit_linear_valid = true;
      impl_->fit_sine_valid = false;
      impl_->fit_speed = linear->speed;
      impl_->fit_linear_cost = linear->cost;
    }
  }
  return inactive_corrected > 0;
}

bool TargetEstimator::diverged() const noexcept
{
  const auto& state = impl_->posterior;
  double face_angle = std::numeric_limits<double>::quiet_NaN();
  if (std::hypot(state[kX], state[kY]) > 0.5) {
    face_angle = std::abs(normalizeAngle(
      state[kPsi] - std::atan2(state[kY], state[kX])));
  }
  return impl_->covariance(kX, kX) > 150.0 ||
         impl_->covariance(kY, kY) > 150.0 ||
         std::abs(state[kX]) > 15.0 || std::abs(state[kY]) > 15.0 ||
         std::abs(state[kZ]) > 5.0 || std::abs(state[kW]) > 10.0 * std::numbers::pi ||
         face_angle > impl_->config.diverge_face_angle_deg * std::numbers::pi / 180.0 ||
         !state.allFinite();
}

State TargetEstimator::state() const noexcept
{
  return impl_->state();
}

const Addition& TargetEstimator::addition() const noexcept
{
  return impl_->addition;
}

}  // namespace L3Estimation::Buff
