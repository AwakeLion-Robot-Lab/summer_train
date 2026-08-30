// UVL 图像观测的正确性与闭环收敛检查。
//
// 四件事，重要性递增：
//   1. 手写投影与 cv::projectPoints 逐点对齐 —— 保证针孔+畸变没写错
//   2. 观测量的几何含义自洽（中心在两端点中点、长度为端点距离、竖直灯条角度近零）
//   3. 观测对误差状态的 Jacobian，Jet 对拍中心差分 —— 验证整条
//      状态 → armorPose → 投影 → [α, uc, vc, L] 的链
//   4. 把 UVL 喂进 ESEKF，从偏离真值的初值出发看是否收敛
//
// 第 4 条是这条路线真正的验收：观测不再是 PnP 解出的位姿，而是图像平面上的
// 灯条几何量，滤波器要靠重投影残差反解整车状态。

#include "l3_estimation/armor/uvl_measure.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l3_estimation/error_state_ekf.hpp"

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <iostream>
#include <numbers>
#include <string_view>
#include <vector>

namespace VM = L3Estimation::VehicleModel;

namespace {

int failure_count = 0;

void expect(bool condition, std::string_view message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failure_count;
  }
}

void expectNear(double actual, double expected, double tolerance, std::string_view message)
{
  if (!(std::abs(actual - expected) <= tolerance)) {
    std::cerr << "FAIL: " << message << "  actual=" << actual << " expected=" << expected
              << " diff=" << std::abs(actual - expected) << '\n';
    ++failure_count;
  }
}

constexpr auto kName = L3Estimation::ArmorName::Infantry3;
constexpr int kArmorNum = 4;
using State = Eigen::Matrix<double, VM::kStateSize, 1>;

cv::Mat makeCameraMatrix()
{
  return (cv::Mat_<double>(3, 3) << 1210.0, 0.0, 721.5, 0.0, 1208.0, 539.5, 0.0, 0.0, 1.0);
}

cv::Mat makeDistortion()
{
  return (cv::Mat_<double>(1, 5) << -0.12, 0.03, 0.0004, -0.0002, 0.0);
}

// 相机光学系（x 右 / y 下 / z 前）在世界系（x 前 / y 左 / z 上）下的位姿。
// 这个纯轴变换矩阵与 CLAUDE.md 记的 T_barrel_camera 姿态部分一致。
Eigen::Isometry3d makeCameraPose()
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = Eigen::Matrix3d{{0.0, 0.0, 1.0}, {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}};
  pose.translation() = Eigen::Vector3d{0.02, -0.01, 0.06};
  return pose;
}

State makeTruth()
{
  State x = State::Zero();
  x[VM::idx::CX] = 3.0;
  x[VM::idx::CY] = 0.45;
  x[VM::idx::CZ] = 0.10;
  x[VM::idx::VCX] = 0.4;
  x[VM::idx::VCY] = -0.2;
  x[VM::idx::ROT_Z] = 0.30;
  x[VM::idx::VYAW] = 2.0;
  x[VM::idx::LOG_R1] = std::log(0.26);
  x[VM::idx::LOG_R2] = std::log(0.285);
  x[VM::idx::HEIGHT] = 0.05;
  return x;
}

L3Estimation::UvlContext makeContext(int id, bool is_left, const Eigen::Isometry3d & camera)
{
  L3Estimation::UvlContext ctx;
  ctx.armor_num = kArmorNum;
  ctx.id = id;
  ctx.is_left = is_left;
  ctx.name = kName;
  ctx.camera_in_world = camera;
  ctx.camera_matrix = makeCameraMatrix();
  ctx.distortion_coefficients = makeDistortion();
  return ctx;
}

// 板是否朝向相机。判据与 awakening 的 match_armor 一致：板的 x 轴指向车心，
// 所以朝外的法向是 -axis_x，与"板 → 相机"方向点积越大越正对。
bool armorFacesCamera(const State & x, int id, const Eigen::Isometry3d & camera)
{
  const auto pose_in_world = VM::armorPose<double>(x.data(), id, kArmorNum, kName);
  const Eigen::Isometry3d pose_in_camera = camera.inverse() * pose_in_world;
  const Eigen::Vector3d front_normal = -pose_in_camera.linear().col(0);
  return front_normal.dot(-pose_in_camera.translation()) > 0.0;
}

}  // namespace

int main()
{
  const Eigen::Isometry3d camera = makeCameraPose();
  const State truth = makeTruth();

  // --- 1. 手写投影 vs cv::projectPoints ------------------------------
  {
    const auto ctx = makeContext(0, true, camera);
    const auto pose_in_world = VM::armorPose<double>(truth.data(), 0, kArmorNum, kName);
    const Eigen::Isometry3d pose_in_camera = camera.inverse() * pose_in_world;

    const std::vector<cv::Point3f> object_points =
      L3Estimation::armorLightPoints3D(kName, true, ctx.armor_config);

    std::vector<Eigen::Vector2d> ours;
    L3Estimation::projectPoints<double>(
      object_points, pose_in_camera, ctx.camera_matrix, ctx.distortion_coefficients, ours);

    cv::Mat rvec;
    cv::Mat rotation(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        rotation.at<double>(r, c) = pose_in_camera.linear()(r, c);
      }
    }
    cv::Rodrigues(rotation, rvec);
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << pose_in_camera.translation().x(),
                    pose_in_camera.translation().y(), pose_in_camera.translation().z());
    std::vector<cv::Point2f> reference;
    cv::projectPoints(
      object_points, rvec, tvec, ctx.camera_matrix, ctx.distortion_coefficients, reference);

    expect(ours.size() == reference.size(), "投影点数量不一致");
    for (std::size_t i = 0; i < ours.size(); ++i) {
      // 物点是 cv::Point3f，两条路径的 float 窄化位置不同，差异在 1e-4 px 量级。
      expectNear(ours[i].x(), reference[i].x, 1e-3, "投影 u 与 cv::projectPoints 不符");
      expectNear(ours[i].y(), reference[i].y, 1e-3, "投影 v 与 cv::projectPoints 不符");
    }
  }

  // --- 2. 观测量的几何含义 -------------------------------------------
  {
    const L3Estimation::UvlMeasure measure{makeContext(0, true, camera)};
    const auto [top, bottom] = measure.projectedPoints(truth);

    L3Estimation::UvlVector z;
    measure(truth.data(), z.data());

    // projectedPoints 返回 cv::Point2f，量级 500 时 float 精度约 3e-5。
    expectNear(
      z[L3Estimation::uvl::CENTER_X], (top.x + bottom.x) / 2.0, 1e-3, "中心 x 不是端点中点");
    expectNear(
      z[L3Estimation::uvl::CENTER_Y], (top.y + bottom.y) / 2.0, 1e-3, "中心 y 不是端点中点");
    expectNear(
      z[L3Estimation::uvl::LENGTH], cv::norm(top - bottom), 1e-3, "长度不是端点距离");

    // 角度落在 ±π 附近而不是 0，这是 awakening 的既有行为，不是笔误：
    // armorLightPoints3D 的第一个点是世界 +z（上），而相机光学系 y 轴朝下，
    // 所以投影后"上端点"的 v 更小、Δy < 0，atan2(Δx, Δy) 自然落在 ±π。
    //
    // 关键在于预测与观测同号、残差走 normalizeAngle，所以滤波器不受影响。但
    // 这也意味着 **角度观测常年贴着 ±π 的分支切口**：求 H 的中心差分必须差
    // residual 而不是 z_pred，否则扰动一跨过切口就会差出 2π 的假梯度。那不是
    // 防御性写法，是必需的。
    expect(
      std::numbers::pi - std::abs(z[L3Estimation::uvl::ANGLE]) < 0.35,
      "竖直灯条的角度观测应当落在 ±π 附近，检查 atan2 的参数顺序与 3D 点序");

    // 预测与观测共用 pointsToObservation，喂同样的点必须得到同样的四维量。
    const L3Estimation::UvlVector from_pixels = L3Estimation::uvlMeasurementFrom(top, bottom);
    expect((from_pixels - z).cwiseAbs().maxCoeff() < 1e-3, "预测与观测的构造不一致");
  }

  // --- 3. 观测 Jacobian：Jet 对拍中心差分 ----------------------------
  //
  // 验证整条 状态 → armorPose → 投影 → [α, uc, vc, L] 的链可微且导数正确。
  {
    using Jet = ceres::Jet<double, VM::kStateSize>;
    const L3Estimation::UvlMeasure measure{makeContext(1, false, camera)};

    std::array<Jet, VM::kStateSize> x_jet;
    for (int i = 0; i < VM::kStateSize; ++i) {
      x_jet[i] = Jet(truth[i], i);
    }
    std::array<Jet, L3Estimation::kUvlMeasureSize> z_jet;
    measure(x_jet.data(), z_jet.data());

    constexpr double kStep = 1e-7;
    double max_error = 0.0;
    for (int col = 0; col < VM::kStateSize; ++col) {
      State plus = truth;
      State minus = truth;
      plus[col] += kStep;
      minus[col] -= kStep;

      L3Estimation::UvlVector z_plus;
      L3Estimation::UvlVector z_minus;
      measure(plus.data(), z_plus.data());
      measure(minus.data(), z_minus.data());

      for (int row = 0; row < L3Estimation::kUvlMeasureSize; ++row) {
        const double numeric = (z_plus[row] - z_minus[row]) / (2.0 * kStep);
        // 像素量级在 1e3，相对误差比绝对误差更有意义。
        const double scale = std::max(1.0, std::abs(numeric));
        max_error = std::max(max_error, std::abs(z_jet[row].v[col] - numeric) / scale);
      }
    }
    expect(max_error < 1e-5, "UVL 观测的 Jet Jacobian 与中心差分不符");
    if (max_error >= 1e-5) {
      std::cerr << "  max relative error = " << max_error << '\n';
    }
  }

  // --- 4. 闭环：用 UVL 观测驱动 ESEKF --------------------------------
  {
    using Filter = L3Estimation::ErrorStateEkf<VM::kStateSize, VM::Motion>;

    State initial = truth;
    initial[VM::idx::CX] += 0.12;
    initial[VM::idx::CY] -= 0.08;
    initial[VM::idx::ROT_Z] += 0.20;
    initial[VM::idx::VYAW] = 0.0;
    initial[VM::idx::VCX] = 0.0;
    initial[VM::idx::VCY] = 0.0;
    initial[VM::idx::LOG_R1] = std::log(0.25);
    initial[VM::idx::LOG_R2] = std::log(0.25);
    initial[VM::idx::HEIGHT] = 0.0;

    const auto inject = [](const auto & delta, auto & nominal) {
      VM::injectState(delta, nominal);
    };
    const auto box_minus = [](const auto & nominal, const auto & value, auto & delta) {
      VM::boxMinusState(nominal, value, delta);
    };

    Eigen::Matrix<double, VM::kStateSize, VM::kStateSize> p0;
    p0.setZero();
    p0.diagonal().setConstant(1.0);
    p0.diagonal()[VM::idx::VCX] = p0.diagonal()[VM::idx::VCY] = 10.0;
    p0.diagonal()[VM::idx::VYAW] = 100.0;

    Eigen::Matrix<double, VM::kStateSize, VM::kStateSize> q;
    q.setZero();
    q.diagonal().setConstant(1e-7);
    q.diagonal()[VM::idx::VCX] = q.diagonal()[VM::idx::VCY] = 1e-2;
    q.diagonal()[VM::idx::VYAW] = 1e-1;

    constexpr double kDt = 0.005;
    Filter filter(
      VM::Motion{.dt = kDt, .name = kName}, [&]() { return q; }, inject, box_minus, p0);
    filter.setState(initial);
    filter.setIterationNum(5);

    // R 按 awakening 的写法：位置与长度的 sigma 正比于灯条像素长度，角度取常数。
    // 除以 2 是因为一块板拆成两条灯条、信息量翻倍。
    constexpr double kSigmaPixelRatio = 0.2;
    constexpr double kSigmaLengthRatio = 0.5;
    constexpr double kSigmaAngle = 0.1;

    State truth_now = truth;
    const VM::Motion truth_motion{.dt = kDt, .name = kName};

    for (int step = 0; step < 600; ++step) {
      State next;
      truth_motion(truth_now.data(), next.data());
      truth_now = next;

      filter.predict();

      std::vector<std::shared_ptr<Filter::ObsBase>> observations;
      for (int id = 0; id < kArmorNum; ++id) {
        if (!armorFacesCamera(truth_now, id, camera)) {
          continue;
        }
        for (const bool is_left : {true, false}) {
          const L3Estimation::UvlMeasure measure{makeContext(id, is_left, camera)};
          const auto [top, bottom] = measure.projectedPoints(truth_now);
          const L3Estimation::UvlVector z = L3Estimation::uvlMeasurementFrom(top, bottom);

          const double length = cv::norm(top - bottom);
          const double sigma_pixel = kSigmaPixelRatio * length;
          const double sigma_length = kSigmaLengthRatio * length;

          Eigen::Matrix<double, L3Estimation::kUvlMeasureSize, L3Estimation::kUvlMeasureSize>
            r_cov;
          r_cov.setZero();
          r_cov(L3Estimation::uvl::ANGLE, L3Estimation::uvl::ANGLE) =
            kSigmaAngle * kSigmaAngle / 2.0;
          r_cov(L3Estimation::uvl::CENTER_X, L3Estimation::uvl::CENTER_X) =
            sigma_pixel * sigma_pixel / 2.0;
          r_cov(L3Estimation::uvl::CENTER_Y, L3Estimation::uvl::CENTER_Y) =
            sigma_pixel * sigma_pixel / 2.0;
          r_cov(L3Estimation::uvl::LENGTH, L3Estimation::uvl::LENGTH) =
            sigma_length * sigma_length / 2.0;

          observations.push_back(Filter::makeObs<L3Estimation::kUvlMeasureSize>(
            z, measure, [r_cov](const L3Estimation::UvlVector &) { return r_cov; },
            [](const L3Estimation::UvlVector & z_pred, const L3Estimation::UvlVector & z_obs) {
              return L3Estimation::UvlMeasure::residual<double>(z_pred, z_obs);
            }));
        }
      }
      if (!observations.empty()) {
        filter.updateMulti(observations);
      }
    }

    const State estimate = filter.state();

    // 编号相位可能与真值差一格，所以断言写在装甲板集合上。
    double max_armor_error = 0.0;
    for (int id = 0; id < kArmorNum; ++id) {
      const auto truth_pose = VM::armorPose<double>(truth_now.data(), id, kArmorNum, kName);
      double best = std::numeric_limits<double>::max();
      for (int other = 0; other < kArmorNum; ++other) {
        const auto estimate_pose =
          VM::armorPose<double>(estimate.data(), other, kArmorNum, kName);
        best =
          std::min(best, (truth_pose.translation() - estimate_pose.translation()).norm());
      }
      max_armor_error = std::max(max_armor_error, best);
    }
    expect(max_armor_error < 0.02, "UVL 闭环下装甲板位置未收敛");
    if (max_armor_error >= 0.02) {
      std::cerr << "  max_armor_error = " << max_armor_error << " m\n";
    }

    expectNear(estimate[VM::idx::CX], truth_now[VM::idx::CX], 0.02, "UVL 闭环下车心 x 未收敛");
    expectNear(estimate[VM::idx::CY], truth_now[VM::idx::CY], 0.02, "UVL 闭环下车心 y 未收敛");
    expectNear(
      std::abs(estimate[VM::idx::VYAW]), std::abs(truth[VM::idx::VYAW]), 0.3,
      "UVL 闭环下角速度未收敛");
  }

  if (failure_count != 0) {
    std::cerr << "uvl measure smoke test failed with " << failure_count << " error(s)\n";
    return 1;
  }
  std::cout << "uvl measure smoke test passed\n";
  return 0;
}
