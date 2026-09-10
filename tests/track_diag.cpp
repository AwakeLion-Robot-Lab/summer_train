// 整车跟踪链路的离线诊断：无显示器，逐帧把 L2 识别、单板 PnP、EKF 内部量
// 和开环预测误差写成 CSV，用来定位"整车预测被什么带偏"。
//
// 它和 auto_aim_test 的分工：auto_aim_test 是人眼看单帧，这个是把整段录像的
// 数量关系压成表格。诊断结论必须能被列出来的数字支撑，只截图看不出偏差是
// 由观测跳变、yaw 搜索截断还是滤波器过程噪声引起的。
#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/tracker.hpp"
#include "runtime/auto_aim_config.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l4_planning/armor/predictor.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

namespace {

constexpr double kRadToDeg = 180.0 / std::numbers::pi;
constexpr double kDegToRad = std::numbers::pi / 180.0;

// optimize_yaw 的三分搜索实际跑在绝对 yaw ∈ [-pi/2, pi/2] 上。落在边界上说明
// 真值在窗口外，此时输出的不是极小值而是截断值，必须单独统计。
constexpr double kSolverYawBound = std::numbers::pi / 2.0;
constexpr double kBoundEpsilonRad = 0.02;

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{calibration c | config/camera_config.yaml | 相机标定 yaml}"
  "{model m | model/armor_model/yolov5.xml | OpenVINO 装甲板模型}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{enemy | blue | 敌方颜色：red / blue / any}"
  "{convention | imu | 录像四元数约定：imu / sp}"
  "{serial-config | config/serial_config.yaml | convention=imu 时读 R_imu_barrel}"
  "{predict-time p | 0.1 | 开环预测时长（秒）}"
  "{bullet-speed | 27.0 | 喂给 L4 的弹速；默认与 SP auto_aim_test 一致（m/s）}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示到结尾}"
  "{scan-step | 1.0 | yaw 代价全周扫描步长（度）}"
  "{out o | /tmp/track_diag | CSV 输出目录}"
  "{@input-path | records/3m_run_mid | avi 和 txt 的路径（不含后缀）}";

struct PoseSample {
  double seconds{0.0};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
};

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

const char* stateName(L3Estimation::TrackState state) noexcept
{
  switch (state) {
  case L3Estimation::TrackState::Lost:
    return "lost";
  case L3Estimation::TrackState::Detecting:
    return "detecting";
  case L3Estimation::TrackState::Tracking:
    return "tracking";
  case L3Estimation::TrackState::TempLost:
    return "temp_lost";
  }
  return "unknown";
}

L2Perception::ArmorColor parseEnemyColor(const std::string& value)
{
  if (value == "red") return L2Perception::ArmorColor::Red;
  if (value == "blue") return L2Perception::ArmorColor::Blue;
  if (value == "any") return L2Perception::ArmorColor::Unknown;
  throw std::invalid_argument("enemy 必须是 red、blue 或 any");
}

bool readPose(std::istream& input, PoseSample& sample)
{
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (!(input >> sample.seconds >> w >> x >> y >> z)) {
    return false;
  }
  sample.q = Eigen::Quaterniond{w, x, y, z};
  require(
    std::isfinite(sample.seconds) && sample.q.coeffs().allFinite() &&
      sample.q.squaredNorm() > 1e-12,
    "四元数文本包含非有限值");
  return true;
}

// 与 auto_aim_test 保持同一份约定转换，两个工具的绝对 yaw 必须可比。
Eigen::Quaterniond toWorldBarrelPose(
  const PoseSample& sample, bool sp_convention, const Eigen::Matrix3d& R_imu_barrel)
{
  if (!sp_convention && R_imu_barrel.isIdentity(0.0)) {
    return sample.q;
  }

  Eigen::Matrix3d R_sp_flip = Eigen::Matrix3d::Identity();
  R_sp_flip(0, 0) = -1.0;
  R_sp_flip(1, 1) = -1.0;
  const Eigen::Matrix3d R_world_imu = sample.q.toRotationMatrix();
  const Eigen::Matrix3d R_world_barrel = sp_convention
    ? Eigen::Matrix3d{R_sp_flip.transpose() * R_world_imu * R_sp_flip}
    : Eigen::Matrix3d{R_world_imu * R_imu_barrel};
  return Eigen::Quaterniond(R_world_barrel);
}

// 把 world 系的一个点投到像素。overlay.csv 用它输出整车中心的像素位置，
// 定义与 auto_aim_test 里画十字用的那个完全一致。
std::optional<cv::Point2d> projectWorldPoint(
  const Eigen::Vector3d& point_in_world,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel)
{
  if (!calibration.T_barrel_camera || !point_in_world.allFinite()) {
    return std::nullopt;
  }
  Eigen::Isometry3d T_world_barrel = Eigen::Isometry3d::Identity();
  T_world_barrel.linear() = q_world_barrel.toRotationMatrix();
  const Eigen::Vector3d point_in_camera =
    (T_world_barrel * *calibration.T_barrel_camera).inverse() * point_in_world;
  if (!point_in_camera.allFinite() || point_in_camera.z() <= 1e-6) {
    return std::nullopt;
  }
  std::vector<cv::Point2d> projected;
  try {
    cv::projectPoints(
      std::vector<cv::Point3d>{
        {point_in_camera.x(), point_in_camera.y(), point_in_camera.z()}},
      cv::Vec3d::all(0.0), cv::Vec3d::all(0.0), calibration.camera_matrix,
      calibration.distortion_coefficients, projected);
  } catch (const cv::Exception&) {
    return std::nullopt;
  }
  if (projected.size() != 1 || !std::isfinite(projected[0].x) ||
      !std::isfinite(projected[0].y)) {
    return std::nullopt;
  }
  return projected[0];
}

// 一块装甲板在图像上的"框中心"：四个投影角点的均值。这就是屏幕上看到的
// 那个绿框的中心，和把三维板心单独投一次不完全相等（透视 + 畸变都非线性），
// 但它才是目视对比的那个量，所以两边统一用这个定义。
std::optional<cv::Point2d> armorBoxCenter(
  const L3Estimation::PnpSolver& solver, const Eigen::Vector4d& xyza,
  L3Estimation::ArmorType type, L3Estimation::ArmorName name)
{
  const std::vector<cv::Point2f> corners =
    solver.reproject_armor(xyza.head<3>(), xyza[3], type, name);
  if (corners.size() != 4) {
    return std::nullopt;
  }
  cv::Point2d sum{0.0, 0.0};
  for (const auto& corner : corners) {
    sum.x += corner.x;
    sum.y += corner.y;
  }
  return cv::Point2d{sum.x / 4.0, sum.y / 4.0};
}

// PnpSolver::armor_reprojection_error 是私有的，这里用它公开的 reproject_armor
// 复算同一个代价：四角点像素距离之和，定义必须与求解器内部完全一致。
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

// 全周扫描 yaw 重投影代价。求解器只在 [-90, 90] 内搜，扫全周才能看出真正的
// 极小值是否被这个窗口切掉，以及代价到底有几个坑。
struct YawScan {
  double best_yaw{std::numeric_limits<double>::quiet_NaN()};
  double best_cost{std::numeric_limits<double>::infinity()};
  // 只在求解器窗口内的极小值，用来和上面的全周结果对比。
  double win_best_yaw{std::numeric_limits<double>::quiet_NaN()};
  double win_best_cost{std::numeric_limits<double>::infinity()};
  std::size_t minima_full{0};
  // 次极小值与全局极小值的代价比。接近 1 说明两个坑深度相当，
  // 任何搜索算法都会在它们之间随机跳。
  double second_ratio{std::numeric_limits<double>::quiet_NaN()};
  double second_yaw{std::numeric_limits<double>::quiet_NaN()};
};

YawScan scanYawCost(
  const L3Estimation::PnpSolver& solver, const L3Estimation::Armor& armor, double step_deg)
{
  YawScan scan;
  const int count = static_cast<int>(std::lround(360.0 / step_deg));
  std::vector<double> yaws;
  std::vector<double> costs;
  yaws.reserve(count);
  costs.reserve(count);
  for (int index = 0; index < count; ++index) {
    const double yaw = L6Telemetry::limit_rad(-std::numbers::pi + index * step_deg * kDegToRad);
    const double cost = yawCost(solver, armor, yaw);
    yaws.push_back(yaw);
    costs.push_back(cost);
    if (cost < scan.best_cost) {
      scan.best_cost = cost;
      scan.best_yaw = yaw;
    }
    if (std::abs(yaw) <= kSolverYawBound && cost < scan.win_best_cost) {
      scan.win_best_cost = cost;
      scan.win_best_yaw = yaw;
    }
  }

  // 代价是周期函数，极小值判定要环形取邻居，否则会漏掉跨越 ±180 的那个坑。
  std::vector<std::size_t> minima;
  for (std::size_t index = 0; index < costs.size(); ++index) {
    const double cost = costs[index];
    if (!std::isfinite(cost)) continue;
    const double previous = costs[(index + costs.size() - 1) % costs.size()];
    const double next = costs[(index + 1) % costs.size()];
    if (cost < previous && cost <= next) {
      minima.push_back(index);
    }
  }
  scan.minima_full = minima.size();

  // 次极小值只在离全局极小值足够远的坑里找，避免把同一个坑的平底数成两个。
  double second = std::numeric_limits<double>::infinity();
  for (std::size_t index : minima) {
    if (std::abs(L6Telemetry::limit_rad(yaws[index] - scan.best_yaw)) < 20.0 * kDegToRad) {
      continue;
    }
    if (costs[index] < second) {
      second = costs[index];
      scan.second_yaw = yaws[index];
    }
  }
  if (std::isfinite(second) && scan.best_cost > 1e-9) {
    scan.second_ratio = second / scan.best_cost;
  }
  return scan;
}

// 复刻 TrackedTarget::update 的装甲面关联：按距离取最近的 3 个候选面，
// 代价是"观测射线方位角之差 + 装甲板 yaw 之差"。诊断必须用同一套规则，
// 否则算出来的残差不是滤波器真正吃进去的那一个。
int associateFace(
  const std::vector<Eigen::Vector4d>& faces, const L3Estimation::Armor& armor)
{
  std::vector<std::pair<double, int>> candidates;
  candidates.reserve(faces.size());
  for (std::size_t index = 0; index < faces.size(); ++index) {
    candidates.emplace_back(
      L6Telemetry::xyz2ypd(faces[index].head<3>()).z(), static_cast<int>(index));
  }
  std::sort(candidates.begin(), candidates.end());

  const int count = std::min<int>(3, static_cast<int>(faces.size()));
  int best_id = -1;
  double best_cost = std::numeric_limits<double>::infinity();
  for (int index = 0; index < count; ++index) {
    const Eigen::Vector4d& face = faces[static_cast<std::size_t>(candidates[index].second)];
    const Eigen::Vector3d ypd = L6Telemetry::xyz2ypd(face.head<3>());
    const double cost =
      std::abs(L6Telemetry::limit_rad(armor.ypd_in_world.x() - ypd.x())) +
      std::abs(L6Telemetry::limit_rad(armor.ypr_in_world[0] - face[3]));
    if (cost < best_cost) {
      best_cost = cost;
      best_id = candidates[index].second;
    }
  }
  return best_id;
}

// 一条缓存的开环预测：把 t 时刻的整车状态外推 horizon 秒后的结果。
struct PendingPrediction {
  L3Estimation::TimePoint valid_at{};
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  std::vector<Eigen::Vector4d> armors;
};

double quadWidth(const std::array<cv::Point2f, 4>& c)
{
  return 0.5 * (cv::norm(c[1] - c[0]) + cv::norm(c[2] - c[3]));
}

double quadHeight(const std::array<cv::Point2f, 4>& c)
{
  return 0.5 * (cv::norm(c[3] - c[0]) + cv::norm(c[2] - c[1]));
}

// 一列数的分位数，用于最后的汇总。就地排序，调用方给的是副本。
double percentile(std::vector<double> values, double ratio)
{
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
    std::clamp(ratio, 0.0, 1.0) * static_cast<double>(values.size() - 1));
  return values[index];
}

double mean(const std::vector<double>& values)
{
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  double sum = 0.0;
  for (double value : values) sum += value;
  return sum / static_cast<double>(values.size());
}

}  // namespace

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
  if (cli.get<bool>("help")) {
    cli.printMessage();
    return 0;
  }

  try {
    L6Telemetry::initLogger();

    const std::string input = cli.get<std::string>("@input-path");
    const std::string video_path = input + ".avi";
    const std::string text_path = input + ".txt";
    const std::string calibration_path = cli.get<std::string>("calibration");
    const double predict_time = cli.get<double>("predict-time");
    const double scan_step = cli.get<double>("scan-step");
    const int start_index = cli.get<int>("start-index");
    const int end_index = cli.get<int>("end-index");
    const auto enemy_color = parseEnemyColor(cli.get<std::string>("enemy"));
    const std::string convention = cli.get<std::string>("convention");
    require(convention == "imu" || convention == "sp", "convention 必须是 imu 或 sp");
    const bool sp_convention = convention == "sp";
    const std::filesystem::path out_dir = cli.get<std::string>("out");
    require(cli.check(), "命令行参数解析失败");

    std::filesystem::create_directories(out_dir);

    const Eigen::Matrix3d R_imu_barrel = sp_convention
      ? Eigen::Matrix3d::Identity()
      : L1Sensor::loadSerialConfig(cli.get<std::string>("serial-config")).R_imu_barrel;

    const YAML::Node calibration_yaml = YAML::LoadFile(calibration_path);
    require(
      static_cast<bool>(calibration_yaml["calibration"]),
      calibration_path + " 里没有 calibration: 节点");
    const auto calibration =
      L1Sensor::loadCameraCalibration(calibration_yaml["calibration"], calibration_path);
    require(calibration.barrelExtrinsicsReady(), calibration_path + " 缺少 T_barrel_camera");

    // L2/L3/L4 参数一律从 auto_aim.yaml 读，回放和实机用同一份数值——否则在
    // YAML 里调噪声或精修阈值，这里根本看不出变化。
    const auto runtime_config = runtime::loadAutoAimConfig("config/auto_aim.yaml");

    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = cli.get<std::string>("model");
    model_config.device = cli.get<std::string>("device");
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);
    require(backend->ready(), "OpenVINO 后端未就绪");
    // 模型来自 --model，没有 auto_aim.yaml 的 layout 可依，按输出形状探契约。
    const auto decoder_config =
      L2Perception::armorDecoderConfigFor(L2Perception::probeOutputSpecs(*backend));
    L2Perception::ArmorDetector detector(
      std::move(backend), decoder_config, L2Perception::ImagePreprocessConfig{},
      runtime_config.refiner);
    require(detector.ready(), "ArmorDetector 未就绪");

    const L3Estimation::ArmorConfig & armor_config = runtime_config.armor;
    L3Estimation::Tracker tracker(
      calibration, armor_config, runtime_config.tracker, runtime_config.target);
    require(tracker.ready(), "Tracker 拒绝了该标定");
    L3Estimation::PnpSolver solver(calibration, armor_config);
    require(solver.ready(), "诊断用 PnpSolver 拒绝了该标定");
    const L4Planning::Predictor predictor;
    L4Planning::Planner planner(runtime_config.plan);
    const double bullet_speed = cli.get<double>("bullet-speed");

    cv::VideoCapture video(video_path);
    require(video.isOpened(), "无法打开录像：" + video_path);
    std::ifstream text(text_path);
    require(text.is_open(), "无法打开四元数文本：" + text_path);

    video.set(cv::CAP_PROP_POS_FRAMES, start_index);
    PoseSample skipped;
    for (int index = 0; index < start_index; ++index) {
      require(readPose(text, skipped), "四元数文本在 start-index 之前结束");
    }

    std::ofstream obs_csv(out_dir / "obs.csv");
    obs_csv << "frame,t,det,class_id,name,conf,area,px_w,px_h,aspect,"
               "yaw_raw_deg,yaw_opt_deg,at_bound,cost_raw,cost_opt,"
               "cost_gmin,yaw_gmin_deg,minima_full,second_ratio,second_yaw_deg,"
               "win_gap_deg,dist_m,x,y,z,az_deg,el_deg,usable\n";
    obs_csv << std::fixed;

    std::ofstream frame_csv(out_dir / "frame.csv");
    frame_csv << "frame,t,dt,gimbal_yaw_deg,ndet,nusable,nmatch,state,"
                 "xc,vx,yc,vy,z,vz,yaw_deg,v_yaw,r1,r2,dz,armor_id,multi,jumped,"
                 "updated,nis,face0,face1,res_az_deg,res_el_deg,res_dist,res_yaw_deg,reset\n";
    frame_csv << std::fixed;

    std::ofstream aim_csv(out_dir / "aim.csv");
    aim_csv << "frame,t,plan_valid,plan_armor_id,aim_x,aim_y,aim_z,"
               "cmd_yaw_deg,cmd_pitch_deg,fly_time,before_fire,fire_admissible,"
               "fire_delta_deg,aim_jump\n";
    aim_csv << std::fixed;

    // 叠加层像素位置。目的是和 sp_vision 逐帧比"框画在哪儿"，所以这里只出
    // 像素，不出世界坐标：世界坐标的差会被距离和视角放大或缩小，看不出屏幕上
    // 到底差了多少。det*_u/v 是当帧 L2 检出的板心，作为"框该落在哪"的参照。
    std::ofstream overlay_csv(out_dir / "overlay.csv");
    overlay_csv << "frame,t,state,center_u,center_v,center_ok,"
                   "a0_u,a0_v,a0_ok,a1_u,a1_v,a1_ok,a2_u,a2_v,a2_ok,"
                   "a3_u,a3_v,a3_ok,ndet,det0_u,det0_v,det1_u,det1_v,"
                   "nearest_px,nearest_id\n";
    overlay_csv << std::fixed;

    std::ofstream pred_csv(out_dir / "pred.csv");
    pred_csv << "frame,t,horizon,center_err,armor_err,armor_yaw_err_deg,obs_err\n";
    pred_csv << std::fixed;

    cv::Mat img;
    PoseSample pose;
    const auto t0 = std::chrono::steady_clock::now();
    std::optional<L3Estimation::TimePoint> last_time;
    auto previous_state = L3Estimation::TrackState::Lost;

    std::deque<PendingPrediction> pending;

    std::size_t frames = 0;
    std::size_t det_total = 0;
    std::size_t obs_total = 0;
    std::size_t usable_total = 0;
    std::size_t bound_hits = 0;
    std::size_t outside_window = 0;
    std::size_t bimodal = 0;
    std::size_t resets = 0;
    std::size_t frames_tracking = 0;
    std::size_t frames_with_det = 0;
    std::size_t double_update_frames = 0;
    std::size_t same_face_frames = 0;
    std::vector<double> yaw_jump_deg;
    std::vector<double> raw_yaw_jump_deg;
    std::optional<double> previous_obs_raw_yaw;
    std::vector<double> nis_values;
    std::vector<double> res_dist;
    std::vector<double> res_yaw_deg;
    std::vector<double> res_az_deg;
    std::vector<double> pred_center_err;
    std::vector<double> pred_obs_err;
    std::vector<double> speeds;
    std::vector<double> vyaws;
    std::vector<double> radii;
    std::optional<double> previous_obs_yaw;
    std::optional<Eigen::Vector3d> previous_obs_xyz;
    std::optional<L3Estimation::TrackedTarget> previous_target;
    std::optional<Eigen::Vector3d> last_aim_point;
    int last_aim_armor_id = -1;
    std::vector<double> aim_jumps;
    std::vector<double> switch_jumps;
    std::vector<double> steady_jumps;

    for (int frame_index = start_index;; ++frame_index) {
      if (end_index > 0 && frame_index > end_index) break;
      video.read(img);
      if (img.empty()) break;
      if (!readPose(text, pose)) break;
      if (frames == 0) {
        require(calibration.matchesImageSize(img.size()), "录像分辨率与标定不一致");
      }
      ++frames;

      const auto timestamp =
        t0 + std::chrono::microseconds(static_cast<long long>(pose.seconds * 1e6));
      const Eigen::Quaterniond q_world_barrel =
        toWorldBarrelPose(pose, sp_convention, R_imu_barrel);
      const double gimbal_yaw =
        L6Telemetry::eulers(q_world_barrel.toRotationMatrix(), 2, 1, 0)[0];
      const double dt = last_time ? L6Telemetry::delta_time(timestamp, *last_time) : 0.0;
      last_time = timestamp;

      auto armors = detector.detect(img);
      std::erase_if(armors, [enemy_color](const L2Perception::Armor& armor) {
        return enemy_color != L2Perception::ArmorColor::Unknown && armor.color != enemy_color;
      });
      det_total += armors.size();
      if (!armors.empty()) ++frames_with_det;

      solver.set_R_world_barrel(q_world_barrel);
      const auto target = tracker.track(armors, q_world_barrel, timestamp);
      const auto& observations = tracker.observations();

      // 观测明细。usable 的判据必须和 Tracker::observationUsable 一致，
      // 否则表里"能用"的行和滤波器实际吃进去的对不上。
      std::size_t usable_here = 0;
      std::size_t match_here = 0;
      for (std::size_t index = 0; index < observations.size(); ++index) {
        const auto& armor = observations[index];
        const auto& detection = armors[index];
        ++obs_total;
        const bool pnp_ok = armor.name != L3Estimation::ArmorName::Unknown;
        const bool usable = pnp_ok && armor.xyz_in_world.allFinite();
        if (usable) ++usable_here;
        if (target && usable && armor.name == target->name) ++match_here;

        if (!pnp_ok) continue;

        const YawScan scan = scanYawCost(solver, armor, scan_step);
        const double yaw_opt = armor.ypr_in_world[0];
        const bool at_bound = std::abs(std::abs(yaw_opt) - kSolverYawBound) < kBoundEpsilonRad;
        if (at_bound) ++bound_hits;
        // 全周极小值落在 [-90, 90] 之外，说明求解器根本够不到真解。
        if (std::abs(scan.best_yaw) > kSolverYawBound) ++outside_window;
        if (std::isfinite(scan.second_ratio) && scan.second_ratio < 1.3) ++bimodal;

        obs_csv << frame_index << ',' << pose.seconds << ',' << index << ','
                << armor.class_id << ',' << static_cast<int>(armor.name) << ','
                << armor.confidence << ',' << armor.area << ','
                << quadWidth(detection.corners) << ',' << quadHeight(detection.corners) << ','
                << (quadHeight(detection.corners) > 0.0
                      ? quadWidth(detection.corners) / quadHeight(detection.corners)
                      : 0.0)
                << ',' << armor.yaw_raw * kRadToDeg << ',' << yaw_opt * kRadToDeg << ','
                << (at_bound ? 1 : 0) << ','
                << yawCost(solver, armor, armor.yaw_raw) << ','
                << yawCost(solver, armor, yaw_opt) << ','
                << scan.best_cost << ',' << scan.best_yaw * kRadToDeg << ','
                << scan.minima_full << ',' << scan.second_ratio << ','
                << scan.second_yaw * kRadToDeg << ','
                << L6Telemetry::limit_rad(scan.best_yaw - scan.win_best_yaw) * kRadToDeg << ','
                << armor.xyz_in_world.norm() << ',' << armor.xyz_in_world.x() << ','
                << armor.xyz_in_world.y() << ',' << armor.xyz_in_world.z() << ','
                << armor.ypd_in_world.x() * kRadToDeg << ','
                << armor.ypd_in_world.y() * kRadToDeg << ',' << (usable ? 1 : 0) << '\n';

        // 逐帧的原始观测跳变。这是"污染滤波器输入"的直接度量，
        // 与滤波器无关，只看 PnP 自己。
        if (usable && index == 0) {
          if (previous_obs_raw_yaw) {
            raw_yaw_jump_deg.push_back(
              std::abs(L6Telemetry::limit_rad(armor.yaw_raw - *previous_obs_raw_yaw)) *
              kRadToDeg);
          }
          previous_obs_raw_yaw = armor.yaw_raw;
          if (previous_obs_yaw) {
            yaw_jump_deg.push_back(
              std::abs(L6Telemetry::limit_rad(yaw_opt - *previous_obs_yaw)) * kRadToDeg);
          }
          if (previous_obs_xyz && dt > 1e-6) {
            speeds.push_back((armor.xyz_in_world - *previous_obs_xyz).norm() / dt);
          }
          previous_obs_yaw = yaw_opt;
          previous_obs_xyz = armor.xyz_in_world;
        }
      }
      usable_total += usable_here;
      if (match_here > 1) ++double_update_frames;

      const auto state = tracker.state();
      if (state == L3Estimation::TrackState::Tracking) ++frames_tracking;
      const bool reset = previous_state != L3Estimation::TrackState::Lost &&
        state == L3Estimation::TrackState::Lost;
      if (reset) ++resets;
      previous_state = state;

      // 单步创新量：把上一帧的后验按同一套恒速模型推到本帧曝光时刻，再和本帧
      // 原始 PnP 相减。这就是 EKF 内部 z - h(x_pri)，但只用公开接口重算，
      // 因此和滤波器内部那份可以互相印证。
      double res_az = std::numeric_limits<double>::quiet_NaN();
      double res_el = std::numeric_limits<double>::quiet_NaN();
      double res_dist_value = std::numeric_limits<double>::quiet_NaN();
      double res_yaw_value = std::numeric_limits<double>::quiet_NaN();
      int face_ids[2] = {-1, -1};
      // TempLost 这一帧没有观测进入滤波器，残差无从谈起。
      if (previous_target && target &&
          state != L3Estimation::TrackState::TempLost && dt > 1e-6) {
        const auto prior = predictor.predict(*previous_target, dt);
        const auto prior_armors = predictor.armorPoses(prior);
        std::size_t slot = 0;
        for (const auto& armor : observations) {
          if (armor.name != target->name || !armor.xyz_in_world.allFinite()) continue;
          const int id = associateFace(prior_armors, armor);
          if (id < 0) continue;
          if (slot < 2) face_ids[slot] = id;
          ++slot;

          const Eigen::Vector4d& face = prior_armors[static_cast<std::size_t>(id)];
          const Eigen::Vector3d prior_ypd = L6Telemetry::xyz2ypd(face.head<3>());
          const double az =
            L6Telemetry::limit_rad(armor.ypd_in_world.x() - prior_ypd.x()) * kRadToDeg;
          const double el =
            L6Telemetry::limit_rad(armor.ypd_in_world.y() - prior_ypd.y()) * kRadToDeg;
          const double distance = armor.ypd_in_world.z() - prior_ypd.z();
          const double yaw_error =
            L6Telemetry::limit_rad(armor.ypr_in_world[0] - face[3]) * kRadToDeg;
          // 表里只留最后一次更新的残差，统计量收全部更新。
          res_az = az;
          res_el = el;
          res_dist_value = distance;
          res_yaw_value = yaw_error;
          res_dist.push_back(std::abs(distance));
          res_yaw_deg.push_back(std::abs(yaw_error));
          res_az_deg.push_back(std::abs(az));
        }
        // 两块可见板关联到同一个物理面，说明整车模型这一帧自相矛盾：
        // 同一个面被两组互斥的观测各更新一次。
        if (slot >= 2 && face_ids[0] == face_ids[1]) ++same_face_frames;
      }

      frame_csv << frame_index << ',' << pose.seconds << ',' << dt << ','
                << gimbal_yaw * kRadToDeg << ',' << armors.size() << ',' << usable_here << ','
                << match_here << ',' << stateName(state) << ',';
      if (target) {
        // 内部状态前十一维：[xc, vx, yc, vy, z, vz, yaw, v_yaw, r1, r2-r1, z2-z1]。
        const Eigen::VectorXd tx = target->ekf_x();
        const double nis = target->ekf().last_nis;
        frame_csv << tx[0] << ',' << tx[1] << ','
                  << tx[2] << ',' << tx[3] << ','
                  << tx[4] << ',' << tx[5] << ','
                  << tx[6] * kRadToDeg << ',' << tx[7] << ',' << tx[8]
                  << ',' << tx[8] + tx[9] << ',' << tx[10] << ','
                  << target->last_id << ',' << (target->last_id != 0 ? 1 : 0) << ','
                  << (target->jumped ? 1 : 0) << ','
                  << (state == L3Estimation::TrackState::TempLost ? 0 : 1) << ','
                  << nis << ',';
        nis_values.push_back(nis);
        vyaws.push_back(std::abs(tx[7]));
        radii.push_back(tx[8]);
      } else {
        // 16 个空字段，与上面 target 分支的列数一一对应。
        for (int column = 0; column < 16; ++column) frame_csv << ',';
      }
      frame_csv << face_ids[0] << ',' << face_ids[1] << ',' << res_az << ',' << res_el
                << ',' << res_dist_value << ','
                << res_yaw_value << ',' << (reset ? 1 : 0) << '\n';
      previous_target = target;

      // 叠加层像素位置：整车中心 + 四块板的框心，外加当帧检出的板心作参照。
      {
        overlay_csv << frame_index << ',' << pose.seconds << ','
                    << stateName(state) << ',';
        const auto center_px = target
          ? projectWorldPoint(
              Eigen::Vector3d{
                target->ekf_x()[0], target->ekf_x()[2], target->ekf_x()[4]},
              calibration, q_world_barrel)
          : std::nullopt;
        if (center_px) {
          overlay_csv << center_px->x << ',' << center_px->y << ",1,";
        } else {
          overlay_csv << ",,0,";
        }

        const auto armor_poses = tracker.targetArmorPoses();
        const auto armor_type = target
          ? L3Estimation::armorTypeOf(target->name).value_or(
              L3Estimation::ArmorType::Small)
          : L3Estimation::ArmorType::Small;
        std::array<std::optional<cv::Point2d>, 4> plate_px{};
        for (std::size_t id = 0; id < 4; ++id) {
          if (target && id < armor_poses.size()) {
            plate_px[id] = armorBoxCenter(
              solver, armor_poses[id], armor_type, target->name);
          }
          if (plate_px[id]) {
            overlay_csv << plate_px[id]->x << ',' << plate_px[id]->y << ",1,";
          } else {
            overlay_csv << ",,0,";
          }
        }

        // 检出的板心（像素），最多两块，按 L2 给出的顺序。
        overlay_csv << armors.size() << ',';
        for (std::size_t index = 0; index < 2; ++index) {
          if (index < armors.size()) {
            overlay_csv << armors[index].center.x << ',' << armors[index].center.y << ',';
          } else {
            overlay_csv << ",,";
          }
        }

        // 每个检出的板心，到最近的那块 EKF 板框心的像素距离。这一列就是
        // "框贴不贴板"，取整帧最大值——统计均值会把偶发的大偏差抹平。
        double worst = -1.0;
        int worst_id = -1;
        for (const auto& detection : armors) {
          double best = std::numeric_limits<double>::infinity();
          int best_id = -1;
          for (std::size_t id = 0; id < plate_px.size(); ++id) {
            if (!plate_px[id]) continue;
            const double dx = plate_px[id]->x - detection.center.x;
            const double dy = plate_px[id]->y - detection.center.y;
            const double distance = std::sqrt(dx * dx + dy * dy);
            if (distance < best) {
              best = distance;
              best_id = static_cast<int>(id);
            }
          }
          if (best_id >= 0 && best > worst) {
            worst = best;
            worst_id = best_id;
          }
        }
        if (worst_id >= 0) {
          overlay_csv << worst << ',' << worst_id << '\n';
        } else {
          overlay_csv << ",\n";
        }
      }

      // L4 规划：瞄准点的稳定性只能在这里量，Tracker 自己不产生瞄准点。
      L1Sensor::RobotState robot_state;
      robot_state.bullet_speed = bullet_speed;
      robot_state.mode = L1Sensor::WorkMode::AutoAim;
      robot_state.rpy.yaw = gimbal_yaw;
      robot_state.timestamp = timestamp;
      const auto plan = planner.plan(target, robot_state, timestamp, false);
      const int armor_id = plan.fire ? plan.fire->armor_id : -1;
      const double fire_facing = plan.fire
        ? plan.fire->facingAngle()
        : std::numeric_limits<double>::quiet_NaN();

      double aim_jump = std::numeric_limits<double>::quiet_NaN();
      if (plan.valid() && last_aim_point) {
        aim_jump = (plan.aim.point - *last_aim_point).norm();
        aim_jumps.push_back(aim_jump);
        if (armor_id != last_aim_armor_id) {
          switch_jumps.push_back(aim_jump);
        } else {
          steady_jumps.push_back(aim_jump);
        }
      }
      if (plan.valid()) {
        last_aim_point = plan.aim.point;
        last_aim_armor_id = armor_id;
      } else {
        last_aim_point.reset();
        last_aim_armor_id = -1;
      }

      aim_csv << frame_index << ',' << pose.seconds << ',' << (plan.valid() ? 1 : 0) << ','
              << armor_id << ',' << plan.aim.point.x() << ',' << plan.aim.point.y() << ','
              << plan.aim.point.z() << ',' << plan.aim.yaw * kRadToDeg << ','
              << plan.aim.pitch * kRadToDeg << ',' << plan.timing.fly_time << ','
              << plan.timing.delay.beforeFire() << ','
              << (plan.fireAdmissible() ? 1 : 0) << ','
              << fire_facing * kRadToDeg << ',' << aim_jump << '\n';

      // 开环预测：缓存 t 时刻外推 predict_time 后的整车，等真到那一刻再对账。
      if (target && predict_time > 0.0) {
        PendingPrediction entry;
        entry.valid_at = timestamp +
          std::chrono::microseconds(static_cast<long long>(predict_time * 1e6));
        const auto predicted = predictor.predict(*target, predict_time);
        const Eigen::VectorXd px = predicted.ekf_x();
        entry.center = {px[0], px[2], px[4]};
        entry.yaw = px[6];
        entry.armors = predictor.armorPoses(predicted);
        pending.push_back(std::move(entry));
      }
      while (!pending.empty() && pending.front().valid_at <= timestamp) {
        const PendingPrediction entry = pending.front();
        pending.pop_front();
        if (!target) continue;

        const Eigen::VectorXd tx = target->ekf_x();
        const double center_err =
          (entry.center - Eigen::Vector3d{tx[0], tx[2], tx[4]}).norm();
        pred_center_err.push_back(center_err);

        // 预测的装甲板 vs 本帧原始 PnP：取最近的那块面，避开关联歧义。
        double best_armor_err = std::numeric_limits<double>::quiet_NaN();
        double best_yaw_err = std::numeric_limits<double>::quiet_NaN();
        double obs_err = std::numeric_limits<double>::quiet_NaN();
        for (const auto& armor : observations) {
          if (armor.name == L3Estimation::ArmorName::Unknown) continue;
          for (const Eigen::Vector4d& pose_xyza : entry.armors) {
            const double distance = (pose_xyza.head<3>() - armor.xyz_in_world).norm();
            if (!std::isfinite(obs_err) || distance < obs_err) {
              obs_err = distance;
              best_yaw_err =
                L6Telemetry::limit_rad(pose_xyza[3] - armor.ypr_in_world[0]) * kRadToDeg;
            }
          }
        }
        best_armor_err = obs_err;
        if (std::isfinite(obs_err)) pred_obs_err.push_back(obs_err);

        pred_csv << frame_index << ',' << pose.seconds << ',' << predict_time << ','
                 << center_err << ',' << best_armor_err << ',' << best_yaw_err << ','
                 << obs_err << '\n';
      }
    }

    aim_csv.close();
    obs_csv.close();
    frame_csv.close();
    pred_csv.close();

    std::cout << "\n=== " << input << " ===\n"
              << "帧数                        " << frames << '\n'
              << "有检出的帧                  " << frames_with_det << '\n'
              << "检出总数                    " << det_total << '\n'
              << "PnP 观测总数                " << obs_total << '\n'
              << "通过 usable 门限            " << usable_total << '\n'
              << "Tracking 帧                 " << frames_tracking << '\n'
              << "跟踪丢失/重置次数           " << resets << '\n'
              << "单帧多观测更新的帧          " << double_update_frames << '\n'
              << "两块板关联到同一物理面的帧  " << same_face_frames << '\n'
              << "-- yaw 搜索 --\n"
              << "顶在 ±90° 边界的观测        " << bound_hits << '\n'
              << "全周极小值在 ±90° 外        " << outside_window << '\n'
              << "次极小值代价比 < 1.3 的观测 " << bimodal << '\n'
              << "-- 原始观测跳变（相邻帧，度）--\n"
              << "mean " << mean(yaw_jump_deg) << "  p50 " << percentile(yaw_jump_deg, 0.5)
              << "  p90 " << percentile(yaw_jump_deg, 0.9) << "  p99 "
              << percentile(yaw_jump_deg, 0.99) << "  max "
              << percentile(yaw_jump_deg, 1.0) << '\n'
              << "-- IPPE 原始 yaw 跳变（optimize_yaw 之前，度）--\n"
              << "mean " << mean(raw_yaw_jump_deg) << "  p50 "
              << percentile(raw_yaw_jump_deg, 0.5) << "  p90 "
              << percentile(raw_yaw_jump_deg, 0.9) << "  p99 "
              << percentile(raw_yaw_jump_deg, 0.99) << "  max "
              << percentile(raw_yaw_jump_deg, 1.0) << '\n'
              << "-- 原始观测隐含速度（m/s）--\n"
              << "mean " << mean(speeds) << "  p50 " << percentile(speeds, 0.5) << "  p90 "
              << percentile(speeds, 0.9) << "  max " << percentile(speeds, 1.0) << '\n'
              << "-- 滤波器 --\n"
              << "NIS  mean " << mean(nis_values) << "  p50 " << percentile(nis_values, 0.5)
              << "  p90 " << percentile(nis_values, 0.9) << "  (4 自由度门限 9.488)\n"
              << "v_yaw |mean| " << mean(vyaws) << "  p90 " << percentile(vyaws, 0.9)
              << "  max " << percentile(vyaws, 1.0) << '\n'
              << "r1   mean " << mean(radii) << "  p50 " << percentile(radii, 0.5) << "  max "
              << percentile(radii, 1.0) << '\n'
              << "-- 单步创新 |z - h(x_pri)| --\n"
              << "方位角(度) mean " << mean(res_az_deg) << "  p90 "
              << percentile(res_az_deg, 0.9) << "  max " << percentile(res_az_deg, 1.0) << '\n'
              << "距离(m)    mean " << mean(res_dist) << "  p90 " << percentile(res_dist, 0.9)
              << "  max " << percentile(res_dist, 1.0) << '\n'
              << "板 yaw(度) mean " << mean(res_yaw_deg) << "  p90 "
              << percentile(res_yaw_deg, 0.9) << "  max " << percentile(res_yaw_deg, 1.0)
              << '\n'
              << "-- 开环预测 " << predict_time * 1e3 << " ms --\n"
              << "中心 vs 后验中心 mean " << mean(pred_center_err) << "  p90 "
              << percentile(pred_center_err, 0.9) << "  max "
              << percentile(pred_center_err, 1.0) << " m\n"
              << "装甲板 vs 原始 PnP  mean " << mean(pred_obs_err) << "  p90 "
              << percentile(pred_obs_err, 0.9) << "  max " << percentile(pred_obs_err, 1.0)
              << " m\n"
              << "-- 瞄准点帧间位移（m）--\n"
              << "全部   mean " << mean(aim_jumps) << "  p90 " << percentile(aim_jumps, 0.9)
              << "  max " << percentile(aim_jumps, 1.0) << "  n=" << aim_jumps.size() << '\n'
              << "未换板 mean " << mean(steady_jumps) << "  p90 "
              << percentile(steady_jumps, 0.9) << "  max "
              << percentile(steady_jumps, 1.0) << "  n=" << steady_jumps.size() << '\n'
              << "换板帧 mean " << mean(switch_jumps) << "  p90 "
              << percentile(switch_jumps, 0.9) << "  max "
              << percentile(switch_jumps, 1.0) << "  n=" << switch_jumps.size() << '\n'
              << "CSV 写入 " << out_dir.string() << '\n';

    L6Telemetry::flushLogger();
    return frames > 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "track_diag 失败: " << error.what() << '\n';
    return 1;
  }
}
