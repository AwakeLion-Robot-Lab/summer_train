// 一次性测量装置：回答"PnP 的 yaw 搜索能不能改成初筛+细筛"。
//
// 判据只有两条，都必须用真实录像的代价曲线说话，不能靠"看起来像单峰"：
//   1. 省不省得起 —— 现在 140 次重投影到底花多少时间；
//   2. 敢不敢省   —— 粗栅格挑出来的那个坑，是不是真解所在的坑。
//      粗筛的风险不是"精度差一点"，而是整个跳到另一个坑里去，
//      那正好就是平面 PnP 二义性本身的失效模式，是最不能赌的地方。
//
// 用完即删，不进 xmake.lua 的常规测试集。
#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "runtime/auto_aim_config.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

namespace {

constexpr double kRadToDeg = 180.0 / std::numbers::pi;
constexpr double kDegToRad = std::numbers::pi / 180.0;
// 与 PnpSolver::kYawSearchRangeDegrees 一致：窗口 140 度、整度枚举。
constexpr int kWindowDegrees = 140;
constexpr double kTruthStepDegrees = 0.02;

const std::string kKeys =
  "{help h usage ? | false | 说明}"
  "{calibration c | config/camera_config.yaml | 相机标定 yaml}"
  "{model m | model/armor_model/yolov5.xml | OpenVINO 模型}"
  "{device d | CPU | 推理设备}"
  "{enemy | blue | red / blue / any}"
  "{convention | imu | imu / sp}"
  "{serial-config | config/serial_config.yaml | convention=imu 时读 R_imu_barrel}"
  "{end-index e | 0 | 结束帧，0 表示到结尾}"
  "{@input-path | records/3m_high | avi/txt 路径前缀}";

void require(bool ok, const std::string& message)
{
  if (!ok) throw std::runtime_error(message);
}

double yawCost(
  const L3Estimation::PnpSolver& solver, const L3Estimation::Armor& armor, double yaw)
{
  const std::vector<cv::Point2f> projected =
    solver.reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  if (projected.size() != armor.points.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double cost = 0.0;
  for (std::size_t index = 0; index < armor.points.size(); ++index) {
    cost += cv::norm(armor.points[index] - projected[index]);
  }
  return cost;
}

// 手写重投影：和 reproject_armor 数学等价，但不过 cv::Mat / Rodrigues。
// 用来量"每次代价评估到底贵在哪"——贵在算法还是贵在接口。
struct FastProjector {
  Eigen::Matrix3d R_world_camera;   // = (R_barrel2world * R_camera2barrel)^T 的转置形式
  Eigen::Vector3d t_world_camera;
  double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
  double k1{0.0}, k2{0.0}, p1{0.0}, p2{0.0}, k3{0.0};
  std::array<Eigen::Vector3d, 4> object_points{};

  double cost(
    const Eigen::Vector3d& xyz_in_world, double yaw, double pitch,
    const std::array<cv::Point2f, 4>& observed) const
  {
    const double sy = std::sin(yaw), cy_ = std::cos(yaw);
    const double sp_ = std::sin(pitch), cp = std::cos(pitch);
    Eigen::Matrix3d R_armor_world;
    R_armor_world << cy_ * cp, -sy, cy_ * sp_,
                     sy * cp, cy_, sy * sp_,
                     -sp_, 0.0, cp;
    const Eigen::Matrix3d R = R_world_camera * R_armor_world;
    const Eigen::Vector3d t = R_world_camera * xyz_in_world + t_world_camera;
    double total = 0.0;
    for (std::size_t index = 0; index < 4; ++index) {
      const Eigen::Vector3d point = R * object_points[index] + t;
      const double inv_z = 1.0 / point.z();
      const double a = point.x() * inv_z, b = point.y() * inv_z;
      const double r2 = a * a + b * b;
      const double radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
      const double u = fx * (a * radial + 2.0 * p1 * a * b + p2 * (r2 + 2.0 * a * a)) + cx;
      const double v = fy * (b * radial + p1 * (r2 + 2.0 * b * b) + 2.0 * p2 * a * b) + cy;
      const double du = u - observed[index].x, dv = v - observed[index].y;
      total += std::sqrt(du * du + dv * dv);
    }
    return total;
  }
};

double percentile(std::vector<double> values, double ratio)
{
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(
    std::clamp(ratio, 0.0, 1.0) * static_cast<double>(values.size() - 1))];
}

Eigen::Quaterniond toWorldBarrel(
  const Eigen::Quaterniond& q, bool sp, const Eigen::Matrix3d& R_imu_barrel)
{
  if (sp) {
    const Eigen::Matrix3d flip = Eigen::Vector3d{-1.0, -1.0, 1.0}.asDiagonal();
    return Eigen::Quaterniond{flip * q.toRotationMatrix() * flip};
  }
  return Eigen::Quaterniond{q.toRotationMatrix() * R_imu_barrel};
}

// 每块板一行结果。
struct Sample {
  double truth_yaw{0.0};       // 窗口内 0.02 度稠密扫描的全局极小
  double grid_yaw{0.0};        // 现行 1 度整步枚举的结果
  double parabola_yaw{0.0};    // 在 1 度栅格胜者两侧做抛物线插值（零额外代价）
  std::vector<double> coarse_err_deg;  // 各粗步长下"细筛之后"离真解还差多少
  std::size_t minima_quarter{0};       // 0.25 度分辨率下窗口内的极小值个数
  double basin_deg{0.0};               // 真解所在坑的宽度（代价回升到 1.02 倍之前）
  double distance{0.0};
};

}  // namespace

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, kKeys);
  if (cli.get<bool>("help")) { cli.printMessage(); return 0; }
  const std::vector<double> coarse_steps{2.0, 4.0, 5.0, 7.0, 10.0, 14.0, 20.0, 35.0};

  try {
    L6Telemetry::initLogger();
    const std::string input = cli.get<std::string>("@input-path");
    const std::string convention = cli.get<std::string>("convention");
    const bool sp = convention == "sp";
    const int end_index = cli.get<int>("end-index");
    const std::string enemy = cli.get<std::string>("enemy");
    require(cli.check(), "命令行解析失败");

    const Eigen::Matrix3d R_imu_barrel = sp
      ? Eigen::Matrix3d::Identity()
      : L1Sensor::loadSerialConfig(cli.get<std::string>("serial-config")).R_imu_barrel;

    const std::string calibration_path = cli.get<std::string>("calibration");
    const YAML::Node yaml = YAML::LoadFile(calibration_path);
    const auto calibration =
      L1Sensor::loadCameraCalibration(yaml["calibration"], calibration_path);
    require(calibration.barrelExtrinsicsReady(), "标定缺少 T_barrel_camera");

    const auto runtime_config = runtime::loadAutoAimConfig("config/auto_aim.yaml");
    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config = runtime_config.inference;
    model_config.model_path = cli.get<std::string>("model");
    model_config.device = cli.get<std::string>("device");
    backend->load(model_config);
    require(backend->ready(), "OpenVINO 未就绪");
    const auto decoder_config =
      L2Perception::armorDecoderConfigFor(L2Perception::probeOutputSpecs(*backend));
    L2Perception::ArmorDetector detector(
      std::move(backend), decoder_config, L2Perception::ImagePreprocessConfig{},
      runtime_config.refiner);
    L3Estimation::PnpSolver solver(calibration, runtime_config.armor);
    require(solver.ready(), "PnpSolver 拒绝了该标定");

    cv::VideoCapture video(input + ".avi");
    require(video.isOpened(), "无法打开 " + input + ".avi");
    std::ifstream text(input + ".txt");
    require(text.is_open(), "无法打开 " + input + ".txt");

    std::vector<Sample> samples;
    double search_time_us_sum = 0.0;
    double fast_time_us_sum = 0.0;
    double fast_max_cost_gap = 0.0;
    double single_pnp_us_sum = 0.0;
    std::size_t fast_grid_disagree = 0;
    double fast_grid_disagree_deg = 0.0;
    std::size_t parabola_at_edge = 0;
    std::size_t parabola_clamped = 0;
    std::size_t parabola_degenerate = 0;
    std::size_t parabola_worse = 0;
    std::vector<double> parabola_regression;
    // 让两条计时路径的结果都逃逸出去，否则 -O2 会把整个循环当死代码删掉，
    // 量出来的就是空循环的时间。
    volatile double sink = 0.0;
    std::size_t search_calls = 0;
    cv::Mat img;

    for (int frame_index = 0;; ++frame_index) {
      if (end_index > 0 && frame_index > end_index) break;
      video.read(img);
      if (img.empty()) break;
      double t = 0.0, w = 0.0, x = 0.0, y = 0.0, z = 0.0;
      if (!(text >> t >> w >> x >> y >> z)) break;
      const Eigen::Quaterniond q_world_barrel =
        toWorldBarrel(Eigen::Quaterniond{w, x, y, z}, sp, R_imu_barrel);

      auto detections = detector.detect(img);
      std::erase_if(detections, [&enemy](const L2Perception::Armor& a) {
        if (enemy == "any") return false;
        return a.color != (enemy == "red" ? L2Perception::ArmorColor::Red
                                          : L2Perception::ArmorColor::Blue);
      });
      solver.set_R_world_barrel(q_world_barrel);

      for (const auto& detection : detections) {
        L3Estimation::Armor armor;
        armor.class_id = detection.class_id;
        armor.confidence = detection.confidence;
        armor.center = detection.center;
        armor.area = L6Telemetry::polygonArea(detection.corners);
        armor.points = detection.corners;
        {
          // 生产路径的真实耗时：solvePnP + 140 次 yaw 搜索，全在 single_pnp 里。
          const auto start = std::chrono::steady_clock::now();
          solver.single_pnp(armor);
          const auto stop = std::chrono::steady_clock::now();
          single_pnp_us_sum +=
            std::chrono::duration<double, std::micro>(stop - start).count();
        }
        if (armor.name == L3Estimation::ArmorName::Unknown) continue;

        const double barrel_yaw =
          L6Telemetry::eulers(q_world_barrel.toRotationMatrix(), 2, 1, 0)[0];
        const double yaw0 =
          L6Telemetry::limit_rad(barrel_yaw - kWindowDegrees / 2.0 * kDegToRad);

        // 现行搜索的耗时：完整跑一遍 140 次重投影。
        {
          const auto start = std::chrono::steady_clock::now();
          double best = std::numeric_limits<double>::infinity();
          double best_yaw = 0.0;
          for (int index = 0; index < kWindowDegrees; ++index) {
            const double yaw = L6Telemetry::limit_rad(yaw0 + index * kDegToRad);
            const double cost = yawCost(solver, armor, yaw);
            if (cost < best) { best = cost; best_yaw = yaw; }
          }
          const auto stop = std::chrono::steady_clock::now();
          search_time_us_sum +=
            std::chrono::duration<double, std::micro>(stop - start).count();
          ++search_calls;
          sink = sink + best + best_yaw;
        }

        // 同一次搜索，改用手写重投影。
        {
          // 外参与 reproject_armor 内部同源：都来自 T_barrel_camera 与本帧姿态。
          const Eigen::Matrix3d R_camera_barrel = calibration.T_barrel_camera->linear();
          const Eigen::Vector3d t_camera_barrel =
            calibration.T_barrel_camera->translation();
          const Eigen::Matrix3d R_barrel_world = q_world_barrel.toRotationMatrix();
          FastProjector fast;
          fast.R_world_camera = R_camera_barrel.transpose() * R_barrel_world.transpose();
          fast.t_world_camera = -R_camera_barrel.transpose() * t_camera_barrel;
          const cv::Mat& matrix = calibration.camera_matrix;
          fast.fx = matrix.at<double>(0, 0); fast.fy = matrix.at<double>(1, 1);
          fast.cx = matrix.at<double>(0, 2); fast.cy = matrix.at<double>(1, 2);
          const cv::Mat& dist = calibration.distortion_coefficients;
          const auto coefficient = [&](int index) {
            return index < static_cast<int>(dist.total())
              ? dist.at<double>(index) : 0.0;
          };
          fast.k1 = coefficient(0); fast.k2 = coefficient(1);
          fast.p1 = coefficient(2); fast.p2 = coefficient(3); fast.k3 = coefficient(4);
          const double half_width = 0.5 * (armor.type == L3Estimation::ArmorType::Big
            ? runtime_config.armor.big_width : runtime_config.armor.small_width);
          const double half_height = 0.5 * runtime_config.armor.height;
          fast.object_points = {
            Eigen::Vector3d{0.0, half_width, half_height},
            Eigen::Vector3d{0.0, -half_width, half_height},
            Eigen::Vector3d{0.0, -half_width, -half_height},
            Eigen::Vector3d{0.0, half_width, -half_height}};
          const double pitch = L3Estimation::armorPitchOf(armor.name);

          const auto start = std::chrono::steady_clock::now();
          double best = std::numeric_limits<double>::infinity();
          for (int index = 0; index < kWindowDegrees; ++index) {
            const double yaw = L6Telemetry::limit_rad(yaw0 + index * kDegToRad);
            best = std::min(best, fast.cost(armor.xyz_in_world, yaw, pitch, armor.points));
          }
          const auto stop = std::chrono::steady_clock::now();
          fast_time_us_sum +=
            std::chrono::duration<double, std::micro>(stop - start).count();
          sink = sink + best;
          // 数值一致性：两条路径在同一 yaw 上的代价必须对得上，否则快路写错了。
          const double reference = yawCost(solver, armor, yaw0);
          const double mine = fast.cost(armor.xyz_in_world, yaw0, pitch, armor.points);
          fast_max_cost_gap = std::max(fast_max_cost_gap, std::abs(reference - mine));
        }

        // 真解：窗口内 0.02 度稠密扫描。
        const int truth_count =
          static_cast<int>(std::lround(kWindowDegrees / kTruthStepDegrees));
        std::vector<double> costs(truth_count);
        double truth_cost = std::numeric_limits<double>::infinity();
        int truth_index = 0;
        for (int index = 0; index < truth_count; ++index) {
          const double yaw = L6Telemetry::limit_rad(
            yaw0 + index * kTruthStepDegrees * kDegToRad);
          costs[index] = yawCost(solver, armor, yaw);
          if (costs[index] < truth_cost) { truth_cost = costs[index]; truth_index = index; }
        }
        if (!std::isfinite(truth_cost)) continue;
        const auto yaw_at = [&](double offset_deg) {
          return L6Telemetry::limit_rad(yaw0 + offset_deg * kDegToRad);
        };
        const auto cost_at = [&](double offset_deg) {
          const double clamped = std::clamp(offset_deg, 0.0, kWindowDegrees - 1e-9);
          return costs[static_cast<int>(std::lround(clamped / kTruthStepDegrees))];
        };

        Sample sample;
        sample.distance = armor.xyz_in_world.norm();
        const double truth_offset = truth_index * kTruthStepDegrees;
        sample.truth_yaw = yaw_at(truth_offset);

        // 真解所在坑的宽度：向两侧走到代价超过极小值 1.02 倍为止。
        double left = truth_offset;
        while (left > 0.0 && cost_at(left) < truth_cost * 1.02) left -= kTruthStepDegrees;
        double right = truth_offset;
        while (right < kWindowDegrees && cost_at(right) < truth_cost * 1.02) {
          right += kTruthStepDegrees;
        }
        sample.basin_deg = right - left;

        // 0.25 度分辨率下的极小值个数：比 0.02 度粗，滤掉像素量化的毛刺。
        for (double offset = 0.25; offset < kWindowDegrees - 0.25; offset += 0.25) {
          const double here = cost_at(offset);
          if (here < cost_at(offset - 0.25) && here <= cost_at(offset + 0.25)) {
            ++sample.minima_quarter;
          }
        }

        // 现行 1 度整步。
        double grid_cost = std::numeric_limits<double>::infinity();
        int grid_index = 0;
        for (int index = 0; index < kWindowDegrees; ++index) {
          const double cost = cost_at(index);
          if (cost < grid_cost) { grid_cost = cost; grid_index = index; }
        }
        sample.grid_yaw = yaw_at(grid_index);

        // 抛物线插值：只用栅格胜者及其左右邻居，扫描时顺手缓存即可，零额外代价。
        {
          if (grid_index == 0 || grid_index == kWindowDegrees - 1) {
            ++parabola_at_edge;
            sample.parabola_yaw = sample.grid_yaw;
          } else {
            const double left_cost = cost_at(grid_index - 1);
            const double mid_cost = cost_at(grid_index);
            const double right_cost = cost_at(grid_index + 1);
            const double denominator = left_cost - 2.0 * mid_cost + right_cost;
            double shift = 0.0;
            if (std::abs(denominator) < 1e-12) {
              ++parabola_degenerate;
            } else {
              shift = 0.5 * (left_cost - right_cost) / denominator;
              if (std::abs(shift) > 0.5) { ++parabola_clamped; }
              shift = std::clamp(shift, -0.5, 0.5);
            }
            sample.parabola_yaw = yaw_at(grid_index + shift);
          }
          const double grid_error =
            std::abs(L6Telemetry::limit_rad(sample.grid_yaw - sample.truth_yaw));
          const double parabola_error =
            std::abs(L6Telemetry::limit_rad(sample.parabola_yaw - sample.truth_yaw));
          if (parabola_error > grid_error) {
            ++parabola_worse;
            parabola_regression.push_back((parabola_error - grid_error) * kRadToDeg);
          }
        }

        // 初筛 + 细筛：粗步长挑坑，坑内用真解级稠密扫描收尾。这样测出来的是
        // "粗筛挑错坑"的纯风险，与细筛算法本身的收敛能力无关，是最乐观的估计。
        for (double step : coarse_steps) {
          double coarse_cost = std::numeric_limits<double>::infinity();
          double coarse_offset = 0.0;
          for (double offset = 0.0; offset < kWindowDegrees; offset += step) {
            const double cost = cost_at(offset);
            if (cost < coarse_cost) { coarse_cost = cost; coarse_offset = offset; }
          }
          const double lo = std::max(0.0, coarse_offset - step);
          const double hi = std::min<double>(kWindowDegrees - kTruthStepDegrees,
                                             coarse_offset + step);
          double refined_cost = std::numeric_limits<double>::infinity();
          double refined_offset = coarse_offset;
          for (double offset = lo; offset <= hi; offset += kTruthStepDegrees) {
            const double cost = cost_at(offset);
            if (cost < refined_cost) { refined_cost = cost; refined_offset = offset; }
          }
          sample.coarse_err_deg.push_back(std::abs(
            L6Telemetry::limit_rad(yaw_at(refined_offset) - sample.truth_yaw)) * kRadToDeg);
        }

        samples.push_back(std::move(sample));
      }
    }

    require(!samples.empty(), "没有可用观测");

    std::printf("\n===== yaw 搜索初筛/细筛测量：%s，%zu 块提交的板 =====\n",
                input.c_str(), samples.size());
    std::printf("生产路径 single_pnp 耗时：均值 %.1f us/块\n",
                single_pnp_us_sum / static_cast<double>(samples.size()));
    std::printf("参考：独立的 140 次整度枚举耗时：均值 %.1f us（%zu 次调用），"
                "单次重投影 %.3f us\n",
                search_time_us_sum / static_cast<double>(search_calls), search_calls,
                search_time_us_sum / static_cast<double>(search_calls) / kWindowDegrees);
    std::printf("同样 140 次、改手写重投影：均值 %.1f us，单次 %.3f us，"
                "提速 %.1f 倍（与 cv 路径的代价最大差 %.3e px）\n",
                fast_time_us_sum / static_cast<double>(search_calls),
                fast_time_us_sum / static_cast<double>(search_calls) / kWindowDegrees,
                search_time_us_sum / fast_time_us_sum, fast_max_cost_gap);
    std::printf("  两条路在 140 点栅格上 argmin 不一致：%zu / %zu 块 (%.2f%%)，"
                "最大相差 %.0f 度\n",
                fast_grid_disagree, samples.size(),
                100.0 * static_cast<double>(fast_grid_disagree) /
                  static_cast<double>(samples.size()),
                fast_grid_disagree_deg);

    std::vector<double> basins, minima, grid_err, para_err;
    for (const auto& s : samples) {
      basins.push_back(s.basin_deg);
      minima.push_back(static_cast<double>(s.minima_quarter));
      grid_err.push_back(
        std::abs(L6Telemetry::limit_rad(s.grid_yaw - s.truth_yaw)) * kRadToDeg);
      para_err.push_back(
        std::abs(L6Telemetry::limit_rad(s.parabola_yaw - s.truth_yaw)) * kRadToDeg);
    }
    std::printf("\n代价曲线形状（窗口内，0.25 度分辨率）\n");
    std::printf("  极小值个数   中位 %.0f   p90 %.0f   最大 %.0f\n",
                percentile(minima, 0.5), percentile(minima, 0.9), percentile(minima, 1.0));
    std::size_t multi = 0;
    for (double m : minima) if (m > 1.0) ++multi;
    std::printf("  非单峰的板   %zu / %zu  (%.1f%%)\n", multi, samples.size(),
                100.0 * static_cast<double>(multi) / static_cast<double>(samples.size()));
    std::printf("  真解坑宽度   中位 %.2f 度   p10 %.2f 度   最窄 %.2f 度\n",
                percentile(basins, 0.5), percentile(basins, 0.1), percentile(basins, 0.0));

    std::printf("\n1 度量化本身的误差（对比 0.02 度稠密真解）\n");
    std::printf("  整度枚举     中位 %.3f 度   p95 %.3f 度   最大 %.3f 度\n",
                percentile(grid_err, 0.5), percentile(grid_err, 0.95),
                percentile(grid_err, 1.0));
    std::printf("  抛物线插值   中位 %.3f 度   p95 %.3f 度   最大 %.3f 度\n",
                percentile(para_err, 0.5), percentile(para_err, 0.95),
                percentile(para_err, 1.0));

    std::printf("  栅格胜者落在窗口边界（无法插值）：%zu   分母退化：%zu   "
                "偏移被夹到 ±0.5：%zu   插值反而更差：%zu\n",
                parabola_at_edge, parabola_degenerate, parabola_clamped,
                parabola_worse);
    std::printf("  变差的那些板，倒退幅度：中位 %.3f 度   p95 %.3f 度   最大 %.3f 度\n",
                percentile(parabola_regression, 0.5),
                percentile(parabola_regression, 0.95),
                percentile(parabola_regression, 1.0));

    std::printf("\n初筛 + 细筛（细筛已用真解级稠密扫描，属最乐观估计）\n");
    std::printf("  粗步长  代价次数  误差中位  误差p95   >2度    >10度\n");
    for (std::size_t k = 0; k < coarse_steps.size(); ++k) {
      std::vector<double> errors;
      std::size_t over2 = 0, over10 = 0;
      for (const auto& s : samples) {
        errors.push_back(s.coarse_err_deg[k]);
        if (s.coarse_err_deg[k] > 2.0) ++over2;
        if (s.coarse_err_deg[k] > 10.0) ++over10;
      }
      // 粗筛 n 次 + 黄金分割细化到 0.05 度：区间每轮乘 0.618。
      const double coarse_evals = std::floor(kWindowDegrees / coarse_steps[k]);
      const double golden =
        std::ceil(std::log(0.05 / (2.0 * coarse_steps[k])) / std::log(0.618));
      std::printf("  %5.1f度  %6.0f次  %7.3f度  %7.3f度  %4.1f%%  %4.1f%%\n",
                  coarse_steps[k], coarse_evals + golden,
                  percentile(errors, 0.5), percentile(errors, 0.95),
                  100.0 * static_cast<double>(over2) / static_cast<double>(samples.size()),
                  100.0 * static_cast<double>(over10) / static_cast<double>(samples.size()));
    }
    std::printf("\n");
    L6Telemetry::flushLogger();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "yaw_search_probe 失败：" << error.what() << '\n';
    return 1;
  }
}
