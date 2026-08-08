#include "l3_estimation/pnp_solver.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <array>
#include <optional>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace L3Estimation {
namespace {

// 角点深度小于该值时视为落在相机平面或相机后方。
constexpr double kMinimumCornerDepth = 1e-6;

// 将识别类别映射为实际 PnP 几何尺寸；未知类别不参与求解。映射本身放在
// types.hpp，与 L5 火控共用同一份，避免两处各写一遍后悄悄分叉。
[[nodiscard]] constexpr std::optional<ArmorType>
armorTypeFromClassId(int class_id) noexcept {
  return armorTypeOf(L2Perception::armorClassFromId(class_id));
}

// 装甲板局部坐标系中 x=0，四点顺序对应图像中的左上、右上、右下、左下。
// yaw 搜索每次迭代都要用到，因此提供不分配堆内存的定长版本。
[[nodiscard]] std::array<cv::Point3d, 4> armorPointsArray(
    ArmorType type, const ArmorConfig &config) {
  const double half_width =
      (type == ArmorType::Big ? config.big_width : config.small_width) / 2.0;
  const double half_height = config.height / 2.0;

  return {cv::Point3d{0.0, half_width, half_height},
          cv::Point3d{0.0, -half_width, half_height},
          cv::Point3d{0.0, -half_width, -half_height},
          cv::Point3d{0.0, half_width, -half_height}};
}

[[nodiscard]] std::vector<cv::Point3d> armorPoints(ArmorType type,
                                                   const ArmorConfig &config) {
  const auto points = armorPointsArray(type, config);
  return {points.begin(), points.end()};
}

// 世界系 yaw 到 armor -> world 旋转。装甲板按车辆类别使用固定安装倾角，
// 只有 yaw 是自由量——这正是 yaw 搜索成立的前提。
[[nodiscard]] Eigen::Matrix3d armorRotationInWorld(double yaw, ArmorName name) {
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  const double pitch = name == ArmorName::Outpost
                           ? -15.0 * std::numbers::pi / 180.0
                           : 15.0 * std::numbers::pi / 180.0;
  const double sin_pitch = std::sin(pitch);
  const double cos_pitch = std::cos(pitch);

  return Eigen::Matrix3d{
      {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
      {sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch},
      {-sin_pitch, 0, cos_pitch}};
}

[[nodiscard]] bool
validCalibration(const L1Sensor::CameraCalibration &calibration) {
  // OpenCV 支持的常用畸变参数长度；拒绝形状虽合法但模型含义未知的数组。
  const std::size_t distortion_count =
      calibration.distortion_coefficients.total();
  const bool distortion_count_ok =
      distortion_count == 4 || distortion_count == 5 || distortion_count == 8 ||
      distortion_count == 12 || distortion_count == 14;

  if (calibration.image_size.width <= 0 || calibration.image_size.height <= 0 ||
      calibration.camera_matrix.type() != CV_64FC1 ||
      calibration.camera_matrix.rows != 3 ||
      calibration.camera_matrix.cols != 3 ||
      calibration.distortion_coefficients.type() != CV_64FC1 ||
      calibration.distortion_coefficients.empty() || !distortion_count_ok ||
      !cv::checkRange(calibration.camera_matrix) ||
      !cv::checkRange(calibration.distortion_coefficients) ||
      calibration.camera_matrix.at<double>(0, 0) <= 0.0 ||
      calibration.camera_matrix.at<double>(1, 1) <= 0.0 ||
      !calibration.T_barrel_camera) {
    return false;
  }

  const auto &transform = *calibration.T_barrel_camera;
  const Eigen::Matrix3d rotation = transform.linear();
  // 静态外参必须是有限的刚体变换，旋转块应接近正交且行列式为 1。
  constexpr double kRotationTolerance = 1e-3;
  return transform.matrix().allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                 .norm() <= kRotationTolerance &&
         std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
}

[[nodiscard]] bool validConfig(const ArmorConfig &config) {
  // 几何尺寸和质量门限必须为有限的正值；面积门限允许显式关闭。
  return std::isfinite(config.small_width) && config.small_width > 0.0 &&
         std::isfinite(config.big_width) && config.big_width > 0.0 &&
         std::isfinite(config.height) && config.height > 0.0 &&
         std::isfinite(config.max_reprojection_error) &&
         config.max_reprojection_error > 0.0 &&
         std::isfinite(config.min_area) && config.min_area >= 0.0;
}

void resetPnpOutput(Armor &armor) {
  // 保留 L2 输入字段，只清除由 PnP 生成的派生结果。
  armor.name = ArmorName::Unknown;
  armor.type = ArmorType::Small;
  armor.xyz_in_camera.setZero();
  armor.xyz_in_world.setZero();
  armor.ypr_in_camera.setZero();
  armor.ypr_in_world.setZero();
  armor.ypd_in_world.setZero();
  armor.reprojection_error = std::numeric_limits<double>::infinity();
  armor.quality = {};
}

[[nodiscard]] bool finiteImagePoints(const std::array<cv::Point2f, 4> &points) {
  // 在调用 OpenCV 前拦截 NaN 和无穷像素坐标。
  return std::all_of(points.begin(), points.end(),
                     [](const cv::Point2f &point) {
                       return std::isfinite(point.x) && std::isfinite(point.y);
                     });
}

} // namespace

PnpSolver::PnpSolver(const L1Sensor::CameraCalibration &calibration,
                     ArmorConfig config)
    : calibration_(calibration), config_(config) {
  ready_ = setCalibration(calibration);
}

bool PnpSolver::setCalibration(const L1Sensor::CameraCalibration &calibration) {
  ready_ = false;
  if (!validCalibration(calibration) || !validConfig(config_)) {
    return false;
  }

  // 缓存静态外参，避免逐帧从齐次变换中重复拆分。
  calibration_ = calibration;
  R_camera2barrel_ = calibration.T_barrel_camera->linear();
  t_camera2barrel_ = calibration.T_barrel_camera->translation();
  ready_ = R_camera2barrel_.allFinite() && t_camera2barrel_.allFinite();

  // 展开内参供快速重投影使用。只接受零斜切且畸变为 4 或 5 个系数的常规标定，
  // 其余（8/12/14 系数的有理、薄棱镜、倾斜模型）不在这里复刻，回退到
  // cv::projectPoints，宁可慢也不要偷偷用错的畸变模型。
  intrinsics_ = PinholeIntrinsics{};
  const std::size_t distortion_count =
      calibration_.distortion_coefficients.total();
  const bool distortion_supported =
      distortion_count == 4 || distortion_count == 5;
  const bool no_skew =
      std::abs(calibration_.camera_matrix.at<double>(0, 1)) < 1e-12;
  if (ready_ && distortion_supported && no_skew) {
    const cv::Mat &K = calibration_.camera_matrix;
    const cv::Mat D = calibration_.distortion_coefficients.reshape(1, 1);
    intrinsics_.fx = K.at<double>(0, 0);
    intrinsics_.fy = K.at<double>(1, 1);
    intrinsics_.cx = K.at<double>(0, 2);
    intrinsics_.cy = K.at<double>(1, 2);
    intrinsics_.k1 = D.at<double>(0, 0);
    intrinsics_.k2 = D.at<double>(0, 1);
    intrinsics_.p1 = D.at<double>(0, 2);
    intrinsics_.p2 = D.at<double>(0, 3);
    intrinsics_.k3 = distortion_count == 5 ? D.at<double>(0, 4) : 0.0;
    intrinsics_.usable = true;
  }
  return ready_;
}

bool PnpSolver::ready() const noexcept { return ready_; }

void PnpSolver::set_R_world_barrel(
    const std::optional<Eigen::Quaterniond> &barrel_pose) {
  // 每帧先清除有效标志，缺失或退化四元数不能沿用上一帧姿态。
  world_barrel_ready_ = false;
  if (!barrel_pose || !barrel_pose->coeffs().allFinite() ||
      barrel_pose->squaredNorm() <= 1e-12) {
    return;
  }

  // 输入四元数可能存在小幅数值漂移，转换前统一归一化。
  R_barrel2world_ = barrel_pose->normalized().toRotationMatrix();
  world_barrel_ready_ = R_barrel2world_.allFinite();
}

void PnpSolver::single_pnp(Armor &armor) const {
  // Armor 可能被跨帧复用；任何提前返回都只能留下明确的无效输出。
  resetPnpOutput(armor);

  const auto armor_type = armorTypeFromClassId(armor.class_id);
  // PnP 同时依赖静态标定、曝光时刻姿态、有效类别和有限角点。
  if (!ready_ || !world_barrel_ready_ || !armor_type ||
      !finiteImagePoints(armor.points)) {
    return;
  }

  const auto object_points = armorPoints(*armor_type, config_);
  const std::vector<cv::Point2f> image_points(armor.points.begin(),
                                              armor.points.end());

  cv::Vec3d rvec;
  cv::Vec3d tvec;
  bool solved = false;
  // 平面 IPPE 使用四个有序角点恢复 armor -> camera 位姿。
  try {
    solved =
        cv::solvePnP(object_points, image_points, calibration_.camera_matrix,
                     calibration_.distortion_coefficients, rvec, tvec, false,
                     cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception &error) {
    L6Telemetry::logWarn("PnpSolver solvePnP failed", error.what());
    return;
  }

  if (!solved) {
    return;
  }
  // pnp_ok 只表示求解器成功返回，后续仍需数值和几何检查。
  armor.quality.pnp_ok = true;

  // 板中心必须在相机前方，旋转向量和平移向量均不得含非有限值。
  if (!std::isfinite(rvec[0]) || !std::isfinite(rvec[1]) ||
      !std::isfinite(rvec[2]) || !std::isfinite(tvec[0]) ||
      !std::isfinite(tvec[1]) || !std::isfinite(tvec[2]) ||
      tvec[2] <= kMinimumCornerDepth) {
    return;
  }

  cv::Matx33d R_armor2camera_cv;
  std::vector<cv::Point2d> reprojected_points;
  // 同时生成旋转矩阵和重投影点，供坐标变换及像素误差计算复用。
  try {
    cv::Rodrigues(rvec, R_armor2camera_cv);
    cv::projectPoints(object_points, rvec, tvec, calibration_.camera_matrix,
                      calibration_.distortion_coefficients, reprojected_points);
  } catch (const cv::Exception &error) {
    L6Telemetry::logWarn("PnpSolver pose conversion failed", error.what());
    return;
  }

  if (!cv::checkRange(cv::Mat(R_armor2camera_cv)) ||
      reprojected_points.size() != image_points.size()) {
    return;
  }

  // 中心在相机前方并不足够，倾斜时四个物理角点也必须全部可见。
  for (const cv::Point3d &point : object_points) {
    const cv::Vec3d point_in_camera =
        R_armor2camera_cv * cv::Vec3d{point.x, point.y, point.z} + tvec;
    if (!std::isfinite(point_in_camera[0]) ||
        !std::isfinite(point_in_camera[1]) ||
        !std::isfinite(point_in_camera[2]) ||
        point_in_camera[2] <= kMinimumCornerDepth) {
      return;
    }
  }
  armor.quality.geometry_ok = true;

  // 观测质量指标取四角点二维欧氏误差的 RMSE，衡量的是 IPPE 这一步解得好不好，
  // 因此用相机系原始解算，和之后的 yaw 优化无关。
  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index) {
    const double dx = reprojected_points[index].x - image_points[index].x;
    const double dy = reprojected_points[index].y - image_points[index].y;
    squared_error_sum += dx * dx + dy * dy;
  }
  const double reprojection_error =
      std::sqrt(squared_error_sum / static_cast<double>(image_points.size()));

  // solvePnP 给出 armor -> camera；静态外参再转换到 barrel 和 world。
  const Eigen::Vector3d xyz_in_camera{tvec[0], tvec[1], tvec[2]};
  const Eigen::Vector3d xyz_in_barrel = R_camera2barrel_ * xyz_in_camera + t_camera2barrel_;
  const Eigen::Vector3d xyz_in_world = R_barrel2world_ * xyz_in_barrel;

  const Eigen::Matrix3d R_armor2camera = L6Telemetry::toEigen(R_armor2camera_cv);
  const Eigen::Matrix3d R_armor2barrel = R_camera2barrel_ * R_armor2camera;
  const Eigen::Matrix3d R_armor2world = R_barrel2world_ * R_armor2barrel;

  //注意这个是朝向角
  const Eigen::Vector3d ypr_in_camera = L6Telemetry::eulers(R_armor2camera, 2, 1, 0);
  const Eigen::Vector3d ypr_in_barrel = L6Telemetry::eulers(R_armor2barrel, 2, 1, 0);
  const Eigen::Vector3d ypr_in_world = L6Telemetry::eulers(R_armor2world, 2, 1, 0);
  //注意这个是方位角
  const Eigen::Vector3d ypd_in_world = L6Telemetry::xyz2ypd(xyz_in_world);

  const bool finite = xyz_in_camera.allFinite() && xyz_in_barrel.allFinite() &&
                      xyz_in_world.allFinite() && R_armor2camera.allFinite() &&
                      R_armor2world.allFinite() && ypr_in_camera.allFinite() &&
                      ypr_in_world.allFinite() && ypd_in_world.allFinite() &&
                      std::isfinite(reprojection_error);
  if (!finite) {
    return;
  }

  // 所有派生量通过有限性检查后再一次性提交，避免暴露半成品观测。
  armor.name = L2Perception::armorClassFromId(armor.class_id);
  armor.type = *armor_type;
  //位置
  armor.xyz_in_camera = xyz_in_camera;
  armor.xyz_in_barrel = xyz_in_barrel;
  armor.xyz_in_world = xyz_in_world;
  //姿态
  armor.ypr_in_camera = ypr_in_camera;
  armor.ypr_in_barrel = ypr_in_barrel;
  armor.ypr_in_world = ypr_in_world;

  armor.ypd_in_world = ypd_in_world;
  armor.reprojection_error = reprojection_error;
  armor.quality.finite = true;
  armor.quality.reprojection_ok =
      reprojection_error <= config_.max_reprojection_error;

  // 场上所有车辆都满足 reproject_armor 的固定安装倾角假设，
  // 因此 yaw 优化对每块装甲板都执行。
  optimize_yaw(armor);
}

void PnpSolver::optimize_yaw(Armor &armor) const {
  // 平面四点 PnP 存在二义性，重投影代价在整周有两个极小值，实测间隔在
  // 100 度以上，深度可以相差一个数量级。任何假设单峰的一维搜索（三分、
  // 黄金分割）都会随初值落进其中任意一个坑，落错时代价会放大 10~30 倍，
  // 输出在两个固定角度之间来回跳，直接污染整车 EKF 的 yaw 和半径。
  //
  // 因此分两段：粗网格在整周锁定全局极小所在的谷，高斯牛顿在谷内细化。
  // 粗扫覆盖整周而不是以枪管 yaw 为中心开窗——开窗要求"真值一定在窗内"，
  // 这个前提被云台大角度转动或外参偏差破坏时是静默错误；整周扫描在快速
  // 重投影下只有几微秒，不值得为省这点时间引入一个前提。
  constexpr int kCoarseSteps = 36;  // 10 度
  constexpr double kCoarseStep = 2.0 * std::numbers::pi / kCoarseSteps;
  // 中心差分的步长。太小会被浮点噪声淹没，太大则 Jacobian 失真；
  // 1e-4 rad 下重投影位移约 1e-2 像素，双精度下信噪比足够。
  constexpr double kDifferenceStep = 1e-4;
  // 收敛判据 1e-5 rad ≈ 0.0006 度，远小于角点噪声对应的 yaw 不确定度。
  constexpr double kConvergenceStep = 1e-5;
  constexpr int kMaxIterations = 8;
  constexpr double kMinimumCurvature = 1e-12;

  armor.yaw_raw = armor.ypr_in_world[0];
  armor.yaw_sigma = std::numeric_limits<double>::infinity();

  // 粗扫。代价只依赖 yaw 的 sin/cos，天然以 2*pi 为周期，因此从 -pi 起
  // 均匀取点就覆盖了全部可能。
  std::array<double, kCoarseSteps> costs{};
  int best_index = -1;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int index = 0; index < kCoarseSteps; ++index) {
    const double yaw = -std::numbers::pi + index * kCoarseStep;
    costs[static_cast<std::size_t>(index)] = yaw_squared_cost(armor, yaw);
    if (costs[static_cast<std::size_t>(index)] < best_cost) {
      best_cost = costs[static_cast<std::size_t>(index)];
      best_index = index;
    }
  }
  // 整周都投影不出来（角点跑到相机后方等），保留 IPPE 的 yaw 不动。
  if (best_index < 0) {
    return;
  }

  double yaw = -std::numbers::pi + best_index * kCoarseStep;

  // 抛物线插值：拿极小格和左右两格拟合二次曲线取顶点。这一步把初值从
  // 半格误差（5 度）压到亚度级，高斯牛顿的迭代次数因此从七八次降到两三次。
  // 取模是因为代价的周期性，第 0 格的左邻居是最后一格。
  {
    const double left = costs[static_cast<std::size_t>(
        (best_index - 1 + kCoarseSteps) % kCoarseSteps)];
    const double right =
        costs[static_cast<std::size_t>((best_index + 1) % kCoarseSteps)];
    const double curvature = left - 2.0 * best_cost + right;
    if (std::isfinite(left) && std::isfinite(right) && curvature > 0.0) {
      yaw += 0.5 * kCoarseStep * (left - right) / curvature;
    }
  }

  // 高斯牛顿 + Levenberg-Marquardt 阻尼。待优化量只有 yaw 一个，因此
  // J^T J 和 J^T r 都是标量，正规方程退化成一次除法。
  double cost = yaw_squared_cost(armor, yaw);
  double lambda = 1e-3;
  double curvature = 0.0;
  bool jacobian_valid = false;
  double gradient = 0.0;

  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    if (!jacobian_valid) {
      std::array<double, 8> residual{};
      std::array<double, 8> forward{};
      std::array<double, 8> backward{};
      if (!yaw_residual(armor, yaw, residual) ||
          !yaw_residual(armor, yaw + kDifferenceStep, forward) ||
          !yaw_residual(armor, yaw - kDifferenceStep, backward)) {
        break;
      }
      // 一维参数的 Jacobian 用中心差分：两次重投影就够，比手推解析式稳妥，
      // 而且改了投影模型也不会忘记同步。
      curvature = 0.0;
      gradient = 0.0;
      for (std::size_t index = 0; index < residual.size(); ++index) {
        const double jacobian =
            (forward[index] - backward[index]) / (2.0 * kDifferenceStep);
        curvature += jacobian * jacobian;
        gradient += jacobian * residual[index];
      }
      jacobian_valid = true;
    }
    // 曲率为零说明 yaw 在此处不可观测，再迭代只会放大数值噪声。
    if (!(curvature > kMinimumCurvature)) {
      break;
    }

    const double step = -gradient / (curvature * (1.0 + lambda));
    const double candidate = yaw + step;
    const double candidate_cost = yaw_squared_cost(armor, candidate);
    if (candidate_cost <= cost) {
      yaw = candidate;
      cost = candidate_cost;
      lambda = std::max(lambda * 0.5, 1e-6);
      jacobian_valid = false;
      if (std::abs(step) < kConvergenceStep) {
        break;
      }
    } else {
      // 拒绝这一步，加大阻尼把方向拉回梯度下降。Jacobian 在原点未变，
      // 不必重算。
      lambda *= 4.0;
      if (lambda > 1e6) {
        break;
      }
    }
  }

  if (!std::isfinite(yaw)) {
    return;
  }

  // 收敛点的曲率就是 Fisher 信息：sigma_yaw^2 = sigma_px^2 / (J^T J)。
  // 网格搜索给不出这个量，而它恰好是 EKF 组装观测噪声 R 时最缺的一项。
  if (!jacobian_valid) {
    std::array<double, 8> forward{};
    std::array<double, 8> backward{};
    if (yaw_residual(armor, yaw + kDifferenceStep, forward) &&
        yaw_residual(armor, yaw - kDifferenceStep, backward)) {
      curvature = 0.0;
      for (std::size_t index = 0; index < forward.size(); ++index) {
        const double jacobian =
            (forward[index] - backward[index]) / (2.0 * kDifferenceStep);
        curvature += jacobian * jacobian;
      }
    } else {
      curvature = 0.0;
    }
  }
  if (curvature > kMinimumCurvature) {
    armor.yaw_sigma = config_.corner_noise_px / std::sqrt(curvature);
  }

  armor.ypr_in_world[0] = L6Telemetry::limit_rad(yaw);
}

bool PnpSolver::yaw_residual(const Armor &armor, double yaw,
                             std::array<double, 8> &residual) const {
  std::array<cv::Point2d, 4> projected{};
  if (!project_armor_points(armor.xyz_in_world, yaw, armor.type, armor.name,
                            projected)) {
    return false;
  }
  // 残差定义为"观测 - 重投影"，顺序与 Armor::points 一致。
  for (std::size_t index = 0; index < projected.size(); ++index) {
    residual[2 * index] = armor.points[index].x - projected[index].x;
    residual[2 * index + 1] = armor.points[index].y - projected[index].y;
  }
  return true;
}

double PnpSolver::yaw_squared_cost(const Armor &armor, double yaw) const {
  // 搜索代价必须是平方和而不是距离之和：后者在残差为零处不可导，
  // 高斯牛顿的正规方程在那里没有定义。
  std::array<double, 8> residual{};
  if (!yaw_residual(armor, yaw, residual)) {
    return std::numeric_limits<double>::infinity();
  }
  double cost = 0.0;
  for (double value : residual) {
    cost += value * value;
  }
  return cost;
}

bool PnpSolver::project_armor_points(
    const Eigen::Vector3d &xyz_in_world, double yaw, ArmorType type,
    ArmorName name, std::array<cv::Point2d, 4> &image_points) const {
  if (!ready_ || !world_barrel_ready_ || !xyz_in_world.allFinite() ||
      !std::isfinite(yaw)) {
    return false;
  }

  // 依次应用 world -> barrel 和 barrel -> camera 逆变换。
  const Eigen::Matrix3d R_armor2world = armorRotationInWorld(yaw, name);
  const Eigen::Matrix3d R_armor2camera = R_camera2barrel_.transpose() *
                                         R_barrel2world_.transpose() *
                                         R_armor2world;
  const Eigen::Vector3d t_armor2camera =
      R_camera2barrel_.transpose() *
      (R_barrel2world_.transpose() * xyz_in_world - t_camera2barrel_);
  if (!R_armor2camera.allFinite() || !t_armor2camera.allFinite()) {
    return false;
  }

  const std::array<cv::Point3d, 4> object_points =
      armorPointsArray(type, config_);
  std::array<Eigen::Vector3d, 4> points_in_camera{};
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    const cv::Point3d &point = object_points[index];
    points_in_camera[index] =
        R_armor2camera * Eigen::Vector3d{point.x, point.y, point.z} +
        t_armor2camera;
    // 相机后方的点投影出来是镜像的假点，整块结果作废。
    if (!points_in_camera[index].allFinite() ||
        points_in_camera[index].z() <= kMinimumCornerDepth) {
      return false;
    }
  }

  if (!intrinsics_.usable) {
    // 非常规畸变模型交给 OpenCV，宁可慢也不要在这里复刻错的公式。
    std::vector<cv::Point3d> camera_points;
    camera_points.reserve(points_in_camera.size());
    for (const Eigen::Vector3d &point : points_in_camera) {
      camera_points.emplace_back(point.x(), point.y(), point.z());
    }
    std::vector<cv::Point2d> projected;
    try {
      cv::projectPoints(camera_points, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0),
                        calibration_.camera_matrix,
                        calibration_.distortion_coefficients, projected);
    } catch (const cv::Exception &error) {
      L6Telemetry::logWarn("PnpSolver armor reprojection failed", error.what());
      return false;
    }
    if (projected.size() != image_points.size()) {
      return false;
    }
    std::copy(projected.begin(), projected.end(), image_points.begin());
    return true;
  }

  // Brown-Conrady：径向 k1/k2/k3 加切向 p1/p2，与 cv::projectPoints 在
  // 4/5 系数、零斜切下逐项等价。setCalibration 已经保证只有这种情况才走到这里。
  for (std::size_t index = 0; index < points_in_camera.size(); ++index) {
    const Eigen::Vector3d &point = points_in_camera[index];
    const double x = point.x() / point.z();
    const double y = point.y() / point.z();
    const double r2 = x * x + y * y;
    const double radial =
        1.0 + r2 * (intrinsics_.k1 + r2 * (intrinsics_.k2 + r2 * intrinsics_.k3));
    const double xy = x * y;
    const double x_distorted =
        x * radial + 2.0 * intrinsics_.p1 * xy + intrinsics_.p2 * (r2 + 2.0 * x * x);
    const double y_distorted =
        y * radial + intrinsics_.p1 * (r2 + 2.0 * y * y) + 2.0 * intrinsics_.p2 * xy;
    image_points[index].x = intrinsics_.fx * x_distorted + intrinsics_.cx;
    image_points[index].y = intrinsics_.fy * y_distorted + intrinsics_.cy;
    if (!std::isfinite(image_points[index].x) ||
        !std::isfinite(image_points[index].y)) {
      return false;
    }
  }
  return true;
}

std::vector<cv::Point2f>
PnpSolver::reproject_armor(const Eigen::Vector3d &xyz_in_world, double yaw,
                           ArmorType type, ArmorName name) const {
  // 只是 project_armor_points 的向量封装。求解器内部不走这里，避免逐次
  // 迭代都分配一个 vector；对外接口保持原样，调用方不受影响。
  std::array<cv::Point2d, 4> projected{};
  if (!project_armor_points(xyz_in_world, yaw, type, name, projected)) {
    return {};
  }

  std::vector<cv::Point2f> image_points;
  image_points.reserve(projected.size());
  for (const cv::Point2d &point : projected) {
    image_points.emplace_back(point);
  }
  return image_points;
}

} // namespace L3Estimation
