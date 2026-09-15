// 整车跟踪链路的离线诊断：无显示器，逐帧把 L2 识别、IESKF 内部量、UVL 创新
// 和开环预测误差写成 CSV，用来定位"整车预测被什么带偏"。
//
// 它和 auto_aim_test 的分工：auto_aim_test 是人眼看单帧，这个是把整段录像的
// 数量关系压成表格。诊断结论必须能被列出来的数字支撑，只截图看不出偏差是
// 由观测、关联还是滤波器过程噪声引起的。
#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "runtime/armor_detector_factory.hpp"
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

const std::string kCommandLineKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{calibration c | config/camera_config.yaml | 相机标定 yaml}"
  "{model m |  | 灯条关键点模型，留空用 auto_aim.yaml 的}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{enemy | blue | 敌方颜色：red / blue / any}"
  "{convention | imu | 录像四元数约定：imu / sp}"
  "{serial-config | config/serial_config.yaml | convention=imu 时读 R_imu_barrel}"
  "{predict-time p | 0.1 | 开环预测时长（秒）}"
  "{bullet-speed | 27.0 | 喂给 L4 的弹速；默认与 SP auto_aim_test 一致（m/s）}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示到结尾}"
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

// 一条缓存的开环预测：把 t 时刻的整车状态外推 horizon 秒后的结果。
struct PendingPrediction {
  L3Estimation::TimePoint valid_at{};
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
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
    // YAML 里调噪声或灯条门限，这里根本看不出变化。
    const auto runtime_config = runtime::loadConfig("config/auto_aim.yaml");

    // 检测器与实机同一个工厂组装，只有模型路径和设备允许命令行覆盖。
    runtime::AutoAimConfig detector_config = runtime_config;
    if (const std::string model = cli.get<std::string>("model"); !model.empty()) {
      detector_config.inference.model_path = model;
    }
    detector_config.inference.device = cli.get<std::string>("device");
    const L2Perception::ArmorDetector detector = runtime::makeDetector(detector_config);
    require(detector.ready(), "ArmorDetector 未就绪");

    const L3Estimation::ArmorConfig & armor_config = runtime_config.armor;
    L3Estimation::EskfTracker tracker(
      calibration, armor_config, runtime_config.ieskf_tracker,
      runtime_config.ieskf_target);
    require(tracker.ready(), "EskfTracker 拒绝了该标定");
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
    obs_csv << "frame,t,det,class_id,name,conf,area,px_w,px_h,aspect\n";
    obs_csv << std::fixed;

    std::ofstream frame_csv(out_dir / "frame.csv");
    frame_csv << "frame,t,dt,gimbal_yaw_deg,ndet,nmatch,state,"
                 "xc,vx,yc,vy,z,vz,yaw_deg,v_yaw,r1,r2,dz,armor_id,jumped,multi,"
                 "updated,nis,nis_dof,nlight,res_angle_deg,res_center_px,res_length_px,res_depth_m,reset\n";
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
    pred_csv << "frame,t,horizon,center_err\n";
    pred_csv << std::fixed;

    cv::Mat img;
    PoseSample pose;
    const auto t0 = std::chrono::steady_clock::now();
    std::optional<L3Estimation::TimePoint> last_time;
    auto previous_state = L3Estimation::TrackState::Lost;

    std::deque<PendingPrediction> pending;

    std::size_t frames = 0;
    std::size_t det_total = 0;
    std::size_t resets = 0;
    std::size_t frames_tracking = 0;
    std::size_t frames_with_det = 0;
    std::size_t double_update_frames = 0;
    std::vector<double> nis_values;
    std::vector<double> res_angle_stats;
    std::vector<double> res_center_stats;
    std::vector<double> res_length_stats;
    std::vector<double> pred_center_err;
    std::vector<double> vyaws;
    std::vector<double> radii;
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

      // 与实跑路径一致：网络 ROI + 独立灯条都走一遍，否则诊断出来的
      // 观测维数和实际滤波器吃到的对不上。
      const std::optional<cv::Rect> light_roi =
        tracker.lightRoi(q_world_barrel, timestamp, img.size());
      const cv::Rect net_roi = tracker.netFocusRoi(
        q_world_barrel, timestamp, img.size(), detector.net_aspect_ratio());
      auto detection_frame = detector.detectFrame(img, light_roi, net_roi);
      auto armors = detection_frame.armors;
      std::erase_if(armors, [enemy_color](const L2Perception::Armor& armor) {
        return enemy_color != L2Perception::ArmorColor::Unknown && armor.color != enemy_color;
      });
      det_total += armors.size();
      if (!armors.empty()) ++frames_with_det;

      solver.set_R_world_barrel(q_world_barrel);
      const auto target = tracker.track(
        armors, detection_frame.lights, q_world_barrel, timestamp);
      const auto& observations = tracker.observations();

      // 观测明细。IESKF 的正常更新只吃类别和角点，不跑 PnP，所以这里只记 L2
      // 侧的几何；与目标同类别的观测多于一块，说明本帧是多板同时更新。
      std::size_t match_here = 0;
      for (std::size_t index = 0; index < observations.size(); ++index) {
        const auto& armor = observations[index];
        const auto& detection = armors[index];
        if (target && armor.name == target->name) ++match_here;

        obs_csv << frame_index << ',' << pose.seconds << ',' << index << ','
                << armor.class_id << ',' << static_cast<int>(armor.name) << ','
                << armor.confidence << ',' << armor.area << ','
                << quadWidth(detection.corners) << ',' << quadHeight(detection.corners) << ','
                << (quadHeight(detection.corners) > 0.0
                      ? quadWidth(detection.corners) / quadHeight(detection.corners)
                      : 0.0)
                << '\n';
      }
      if (match_here > 1) ++double_update_frames;

      const auto state = tracker.state();
      if (state == L3Estimation::TrackState::Tracking) ++frames_tracking;
      const bool reset = previous_state != L3Estimation::TrackState::Lost &&
        state == L3Estimation::TrackState::Lost;
      if (reset) ++resets;
      previous_state = state;

      // UVL 创新量。四个观测分量量纲不同（角度 rad、中心和长度 px），
      // 混进一个范数没有意义，所以按物理含义分开取：中心残差大说明整车位置
      // 偏了，长度残差大说明深度偏了，角度残差大说明姿态偏了。
      // UVL 的残差只有滤波器自己算得出（要投影全部灯条端点），所以直接取
      // EskfTarget 暴露的那份。
      double res_angle_deg = std::numeric_limits<double>::quiet_NaN();
      double res_center_px = std::numeric_limits<double>::quiet_NaN();
      double res_length_px = std::numeric_limits<double>::quiet_NaN();
      double res_depth_m = std::numeric_limits<double>::quiet_NaN();
      int res_light_count = 0;
      // TempLost 这一帧没有观测进入滤波器，残差无从谈起。
      if (target && state != L3Estimation::TrackState::TempLost) {
        const auto& residual = target->lastUvlResidual();
        if (residual.light_count > 0) {
          res_angle_deg = residual.angle_rms_deg;
          res_center_px = residual.center_rms_px;
          res_length_px = residual.length_rms_px;
          res_depth_m = residual.depth_diff_m;
          res_light_count = residual.light_count;
          res_angle_stats.push_back(residual.angle_rms_deg);
          res_center_stats.push_back(residual.center_rms_px);
          res_length_stats.push_back(residual.length_rms_px);
        }
      }

      frame_csv << frame_index << ',' << pose.seconds << ',' << dt << ','
                << gimbal_yaw * kRadToDeg << ',' << armors.size() << ','
                << match_here << ',' << stateName(state) << ',';
      if (target) {
        // ekf_x() 布局：[cx, vcx, cy, vcy, cz, vcz, rot_z, vyaw, r1, P1, P2,
        // rot_y, rot_x]，半径已转回线性。姿态是完整 SO(3)，yaw 必须从旋转矩阵
        // 分解，不能直接读 rot_z；P1 在四板车上是线性 r2，前哨站上是 dz1，
        // 所以 r2 列只对四板车有值。dz 列取 P2：四板车为奇偶板高度差，前哨站为 dz2。
        const Eigen::VectorXd tx = target->ekf_x();
        const double nis = target->lastNis();
        const double yaw = L6Telemetry::eulers(
          L3Estimation::VehicleModel::stateRotation(tx), 2, 1, 0)[0];
        const double r2 = target->armor_num() == 4
          ? tx[L3Estimation::VehicleModel::idx::LOG_R2]
          : std::numeric_limits<double>::quiet_NaN();
        frame_csv << tx[0] << ',' << tx[1] << ','
                  << tx[2] << ',' << tx[3] << ','
                  << tx[4] << ',' << tx[5] << ','
                  << yaw * kRadToDeg << ',' << tx[7] << ',' << tx[8]
                  << ',' << r2 << ',' << tx[L3Estimation::VehicleModel::idx::P2] << ','
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
      frame_csv << (target ? target->lastNisDof() : 0) << ',' << res_light_count << ','
                << res_angle_deg << ',' << res_center_px << ',' << res_length_px << ','
                << res_depth_m << ',' << (reset ? 1 : 0) << '\n';

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

        const auto armor_poses = tracker.armorPoses();
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

        pred_csv << frame_index << ',' << pose.seconds << ',' << predict_time << ','
                 << center_err << '\n';
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
              << "Tracking 帧                 " << frames_tracking << '\n'
              << "跟踪丢失/重置次数           " << resets << '\n'
              << "单帧多观测更新的帧          " << double_update_frames << '\n'
              << "-- 滤波器 --\n"
              << "NIS  mean " << mean(nis_values) << "  p50 " << percentile(nis_values, 0.5)
              << "  p90 " << percentile(nis_values, 0.9)
              << "  (UVL 观测维数每帧在变，见 nis_dof 列，固定卡方门限不适用)\n"
              << "v_yaw |mean| " << mean(vyaws) << "  p90 " << percentile(vyaws, 0.9)
              << "  max " << percentile(vyaws, 1.0) << '\n'
              << "r1   mean " << mean(radii) << "  p50 " << percentile(radii, 0.5) << "  max "
              << percentile(radii, 1.0) << '\n'
              << "-- UVL 创新（按分量，量纲不同不能合并）--\n"
              << "角度(度)   mean " << mean(res_angle_stats) << "  p90 "
              << percentile(res_angle_stats, 0.9) << "  max "
              << percentile(res_angle_stats, 1.0) << '\n'
              << "中心(px)   mean " << mean(res_center_stats) << "  p90 "
              << percentile(res_center_stats, 0.9) << "  max "
              << percentile(res_center_stats, 1.0) << '\n'
              << "长度(px)   mean " << mean(res_length_stats) << "  p90 "
              << percentile(res_length_stats, 0.9) << "  max "
              << percentile(res_length_stats, 1.0) << '\n'
              << "-- 开环预测 " << predict_time * 1e3 << " ms --\n"
              << "中心 vs 后验中心 mean " << mean(pred_center_err) << "  p90 "
              << percentile(pred_center_err, 0.9) << "  max "
              << percentile(pred_center_err, 1.0) << " m\n"
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
