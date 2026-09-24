// 整车跟踪链路的离线诊断：无显示器，逐帧把 L2 识别、IESKF 内部量、端点创新
// 和开环预测误差写成 CSV，用来定位"整车预测被什么带偏"。
//
// 它和 auto_aim_test 的分工：auto_aim_test 是人眼看单帧，这个是把整段录像的
// 数量关系压成表格。诊断结论必须能被列出来的数字支撑，只截图看不出偏差是
// 由观测、关联还是滤波器过程噪声引起的。
//
// 衡量 L3 好坏的判据只有一个：pred.csv 的 pred_px —— t 时刻外推 predict_time
// 后的整车四块板，等真到那一刻，用那一帧的云台姿态重投到像素，和那一帧真实
// 检出的板心做一对一匹配。两条纪律来自踩过的坑：
//
//   * 不要拿估计去比估计。"预测中心 vs 后来的后验中心"、"离检出最近的那块板
//     的距离"都是滤波器自己和自己对账——最近的那块板恰好就是当帧正在被观测
//     更新的板，怎么调都好看，整车里没被观测的那三块板一点都没量到。
//   * 一对一分配是必须的。允许两个检出都匹配同一块板的话，整车转过一个板位
//     （yaw 偏 2π/N）在数字上看不出来。
//
// hold_px 是同一时刻完全不外推的基准。预测必须比它准，否则速度项是负贡献。
#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/auto_aim_config.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l4_planning/armor/predictor.hpp"
#include "l6_telemetry/aim_overlay.hpp"
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
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
  "{model m |  | 整板模型，留空用 auto_aim.yaml 的；给了就按输出形状认 layout}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{enemy | blue | 敌方颜色：red / blue / any}"
  "{convention | imu | 录像四元数约定：imu / sp}"
  "{serial-config | config/serial_config.yaml | convention=imu 时读 R_imu_barrel}"
  "{predict-time p | 0.1 | 开环预测时长（秒）}"
  "{bullet-speed | 27.0 | 喂给 L4 的弹速；默认与 SP auto_aim_test 一致（m/s）}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示到结尾}"
  "{config | config/auto_aim.yaml | L2/L3/L4 参数；A/B 时指向改过的副本，不动仓库里那份}"
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

// overlay.csv 的整车中心像素位置直接用叠加层那份投影：实机画十字、
// auto_aim_test 画十字、这里写 CSV 必须是同一个定义，各写一份迟早会漂。
using L6Telemetry::projectWorldPoint;

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

// 一条缓存的开环预测：t 时刻外推 horizon 秒后的整车，存下四块板的 world 位姿，
// 等真到那一刻再对账。见文件头对判据的说明。
struct PendingPrediction {
  L3Estimation::TimePoint made_at{};
  L3Estimation::TimePoint valid_at{};
  // 外推后的四块板，以及同一时刻完全不外推的那份（基准）。
  std::vector<Eigen::Vector4d> armors;
  std::vector<Eigen::Vector4d> armors_hold;
  L3Estimation::ArmorType type{L3Estimation::ArmorType::Small};
  L3Estimation::ArmorName name{L3Estimation::ArmorName::Unknown};
};

// 把若干块板的重投影框心一对一分配给本帧检出的板心，返回最差的那一对的
// 像素距离。一对一是关键：允许两个检出都匹配同一块板，整车转错了也看不出来。
std::optional<double> worstAssignment(
  const std::vector<std::optional<cv::Point2d>>& plates,
  const std::vector<cv::Point2f>& detections)
{
  std::vector<cv::Point2d> usable;
  for (const auto& plate : plates) {
    if (plate) usable.push_back(*plate);
  }
  if (detections.empty() || usable.size() < detections.size()) {
    return std::nullopt;
  }
  // 板最多四块、检出最多两块，直接枚举全排列取前几位即可；同一种分配会被
  // 重复访问几次，但总共不过 24 轮，不值得为此写一套匈牙利。
  std::vector<std::size_t> order(usable.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  double best_sum = std::numeric_limits<double>::infinity();
  double best_worst = std::numeric_limits<double>::infinity();
  do {
    double sum = 0.0;
    double worst = 0.0;
    for (std::size_t i = 0; i < detections.size(); ++i) {
      const cv::Point2d& plate = usable[order[i]];
      const double distance = std::hypot(
        plate.x - detections[i].x, plate.y - detections[i].y);
      sum += distance;
      worst = std::max(worst, distance);
    }
    if (sum < best_sum) {
      best_sum = sum;
      best_worst = worst;
    }
  } while (std::next_permutation(order.begin(), order.end()));
  return best_worst;
}

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

// 落在门限内的比例（百分数）。均值会被少数几帧的大偏差拖走，"贴不贴板"
// 还是按比例看更直观。
double hitRate(const std::vector<double>& values, double threshold)
{
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::size_t hits = 0;
  for (double value : values) {
    if (value <= threshold) ++hits;
  }
  return 100.0 * static_cast<double>(hits) / static_cast<double>(values.size());
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
    const auto runtime_config = runtime::loadConfig(cli.get<std::string>("config"));

    // 检测器与实机同一个工厂组装，只有模型路径和设备允许命令行覆盖。命令行换了
    // 模型时 YAML 里的 layout 未必配得上，按模型输出形状认。
    runtime::AutoAimConfig detector_config = runtime_config;
    const std::string model_override = cli.get<std::string>("model");
    if (!model_override.empty()) {
      detector_config.inference.model_path = model_override;
    }
    detector_config.inference.device = cli.get<std::string>("device");
    const L2Perception::ArmorDetector detector =
      runtime::makeDetector(detector_config, !model_override.empty());
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
    obs_csv << "frame,t,det,class_id,net_class_id,num_conf,name,conf,area,px_w,px_h,aspect\n";
    obs_csv << std::fixed;

    std::ofstream frame_csv(out_dir / "frame.csv");
    frame_csv << "frame,t,dt,gimbal_yaw_deg,ndet,nmatch,state,"
                 "xc,vx,yc,vy,z,vz,yaw_deg,v_yaw,r1,r2,dz,armor_id,jumped,multi,"
                 "updated,nis,nis_dof,nlight,res_along_px,res_perp_px,"
                 "res_shift_perp_mean,res_shift_perp_rms,"
                 "res_shift_along_mean,res_shift_along_rms,"
                 "res_tilt_mean_deg,res_tilt_rms_deg,"
                 "res_len_mean,res_len_rms,res_depth_m,reset\n";
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
    pred_csv << "frame,t,horizon,ndet,pred_px,hold_px\n";
    pred_csv << std::fixed;

    cv::Mat img;
    PoseSample pose;
    const auto t0 = std::chrono::steady_clock::now();
    std::optional<L3Estimation::TimePoint> last_time;
    std::size_t previous_drops = 0;

    std::deque<PendingPrediction> pending;

    std::size_t frames = 0;
    std::size_t det_total = 0;
    std::size_t resets = 0;
    std::size_t frames_tracking = 0;
    std::size_t frames_with_det = 0;
    std::size_t double_update_frames = 0;
    // 侧边灯条这一路真正的产出量。double_update_frames 数的是"同类别装甲板多于
    // 一块"，跟侧边灯条无关，别拿它当灯条检出量看。
    std::size_t side_light_total = 0;
    std::size_t frames_with_side_light = 0;
    // L3 真正吃下去的那些：过了 matchLight 全部门限、进了 updateMulti 的
    // 独立灯条。与上面 L2 的检出量一起看，才知道关联门限收紧了多少。
    std::size_t side_light_used = 0;
    std::size_t number_accepted = 0;
    std::size_t number_dropped = 0;
    std::vector<double> ms_l2;
    // 其中侧边灯条那一路（找灯条 + 判色 + 判重），只记 L3 给了 ROI 的帧。
    std::vector<double> ms_side_light;
    std::vector<double> ms_l3;
    std::vector<double> ms_l4;
    std::vector<double> pred_pixel_err;
    std::vector<double> hold_pixel_err;
    std::vector<double> nis_values;
    std::vector<double> nis_per_dof;
    std::size_t refine_hit = 0;
    std::size_t refine_kept = 0;
    std::size_t refine_no_bar = 0;
    std::size_t refine_too_short = 0;
    std::size_t refine_shift_rej = 0;
    std::size_t refine_contours = 0;
    std::size_t refine_bar_kept = 0;
    std::size_t refine_rej_angle = 0;
    std::size_t refine_rej_ratio = 0;
    std::size_t refine_rej_len = 0;
    std::vector<double> res_along_stats;
    std::vector<double> res_perp_stats;
    // 四个物理通道的逐帧统计，下标与 LightResidual 同序：横移⊥、沿移∥、
    // 倾角、长度。mean 一列用来看系统偏差，rms 一列看噪声。
    std::array<std::vector<double>, 4> res_channel_mean;
    std::array<std::vector<double>, 4> res_channel_rms;
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
      const auto light_hints = tracker.lightHints(q_world_barrel, timestamp);
      const auto t_l2_begin = std::chrono::steady_clock::now();
      // 敌方颜色必须传进去：侧边灯条按它筛色，findLights 的通道相减也只在
      // 颜色已知时才启用。不传等于把这条路默认关掉，而且会把友方灯条一起
      // 喂进滤波器 —— auto_aim_test 一直是传的，两个回放器不能不一致。
      auto detection_frame =
        detector.detectFrame(img, light_roi, net_roi, enemy_color, light_hints);
      const auto t_l2_end = std::chrono::steady_clock::now();
      auto armors = detection_frame.armors;
      std::erase_if(armors, [enemy_color](const L2Perception::Armor& armor) {
        return enemy_color != L2Perception::ArmorColor::Unknown && armor.color != enemy_color;
      });
      det_total += armors.size();
      if (!armors.empty()) ++frames_with_det;
      side_light_total += detection_frame.lights.size();
      if (!detection_frame.lights.empty()) ++frames_with_side_light;

      solver.set_R_world_barrel(q_world_barrel);
      const auto t_l3_begin = std::chrono::steady_clock::now();
      const auto target = tracker.track(
        armors, detection_frame.lights, q_world_barrel, timestamp);
      const auto t_l3_end = std::chrono::steady_clock::now();
      for (const auto& used : tracker.usedLights()) {
        if (used.isolated) ++side_light_used;
      }
      ms_l2.push_back(std::chrono::duration<double, std::milli>(t_l2_end - t_l2_begin).count());
      if (light_roi) ms_side_light.push_back(detector.lastTiming().side_light);
      ms_l3.push_back(std::chrono::duration<double, std::milli>(t_l3_end - t_l3_begin).count());

      // 观测明细。IESKF 的正常更新只吃类别和角点，不跑 PnP，所以这里只记 L2
      // 侧的几何；与目标同类别的观测多于一块，说明本帧是多板同时更新。
      std::size_t match_here = 0;
      for (std::size_t index = 0; index < armors.size(); ++index) {
        const auto& detection = armors[index];
        const auto armor = L3Estimation::toObservation(detection);
        const auto name = L2Perception::armorClassFromId(armor.class_id);
        if (target && name == target->name) ++match_here;

        obs_csv << frame_index << ',' << pose.seconds << ',' << index << ','
                << armor.class_id << ',' << detection.network_class_id << ','
                << detection.number_confidence << ','
                << static_cast<int>(name) << ','
                << armor.confidence << ',' << armor.area << ','
                << quadWidth(detection.corners) << ',' << quadHeight(detection.corners) << ','
                << (quadHeight(detection.corners) > 0.0
                      ? quadWidth(detection.corners) / quadHeight(detection.corners)
                      : 0.0)
                << '\n';
      }
      if (match_here > 1) ++double_update_frames;
      // 数字二次分类丢掉的板：全局统计，判断门限是不是把有效观测也筛掉了。
      // 精修的触发比例。判断"换底图之后精修变好了"还是"只是更少触发了"——
      // 后者等于偷偷关掉功能，两种情况在 pred_px 上长得一模一样。
      refine_hit += detector.lastRefine().refined;
      refine_kept += detector.lastRefine().network_kept;
      refine_no_bar += detector.lastRefine().no_lightbar;
      refine_too_short += detector.lastRefine().size_skipped;
      refine_shift_rej += detector.lastRefine().shift_rejected;
      refine_contours += detector.lastRefine().contour_total;
      refine_bar_kept += detector.lastRefine().bar_kept;
      refine_rej_angle += detector.lastRefine().rej_angle;
      refine_rej_ratio += detector.lastRefine().rej_ratio;
      refine_rej_len += detector.lastRefine().rej_length;
      number_accepted += detector.lastNumbers().accepted;
      number_dropped += detector.lastNumbers().dropped();

      const auto state = tracker.state();
      if (state == L3Estimation::TrackState::Tracking) ++frames_tracking;
      // 丢弃后同一帧就可能重建，状态上看不出来，按跟踪器自己的计数判断。
      const bool reset = tracker.dropCount() != previous_drops;
      if (reset) ++resets;
      previous_drops = tracker.dropCount();

      // 端点创新量，投到每根灯条自己的坐标系里分方向取：沿灯条分量大多半是
      // 深度或高度偏了，垂直分量大是横向位置或姿态偏了。残差只有滤波器自己
      // 算得出（要投影全部灯条端点），所以直接取 EskfTarget 暴露的那份。
      double res_along_px = std::numeric_limits<double>::quiet_NaN();
      double res_perp_px = std::numeric_limits<double>::quiet_NaN();
      double res_depth_m = std::numeric_limits<double>::quiet_NaN();
      // 与 res_channel_* 同序，倾角一列换成度写进 CSV。
      std::array<double, 4> res_channel_mean_row{};
      std::array<double, 4> res_channel_rms_row{};
      res_channel_mean_row.fill(std::numeric_limits<double>::quiet_NaN());
      res_channel_rms_row.fill(std::numeric_limits<double>::quiet_NaN());
      int res_light_count = 0;
      // TempLost 这一帧没有观测进入滤波器，残差和 NIS 都是上一帧留下的，
      // 不进统计。
      if (target && state != L3Estimation::TrackState::TempLost) {
        const auto& residual = target->lastLightResidual();
        if (residual.light_count > 0) {
          res_along_px = residual.along_rms_px;
          res_perp_px = residual.perp_rms_px;
          res_depth_m = residual.depth_diff_m;
          res_light_count = residual.light_count;
          res_along_stats.push_back(residual.along_rms_px);
          res_perp_stats.push_back(residual.perp_rms_px);

          const std::array<const L3Estimation::EskfTarget::LightResidual::Channel *, 4>
            channels{&residual.shift_perp, &residual.shift_along, &residual.tilt,
                     &residual.length};
          for (std::size_t k = 0; k < channels.size(); ++k) {
            // 倾角那一路存成度，CSV 和汇总都按度读。
            const double scale = (k == 2) ? 180.0 / CV_PI : 1.0;
            res_channel_mean_row[k] = channels[k]->mean * scale;
            res_channel_rms_row[k] = channels[k]->rms * scale;
            res_channel_mean[k].push_back(res_channel_mean_row[k]);
            res_channel_rms[k].push_back(res_channel_rms_row[k]);
          }
          nis_values.push_back(target->lastNis());
          if (target->lastNisDof() > 0) {
            nis_per_dof.push_back(target->lastNis() / target->lastNisDof());
          }
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
        vyaws.push_back(std::abs(tx[7]));
        radii.push_back(tx[8]);
      } else {
        // 16 个空字段，与上面 target 分支的列数一一对应。
        for (int column = 0; column < 16; ++column) frame_csv << ',';
      }
      frame_csv << (target ? target->lastNisDof() : 0) << ',' << res_light_count << ','
                << res_along_px << ',' << res_perp_px << ',';
      for (std::size_t k = 0; k < res_channel_mean_row.size(); ++k) {
        frame_csv << res_channel_mean_row[k] << ',' << res_channel_rms_row[k] << ',';
      }
      frame_csv << res_depth_m << ',' << (reset ? 1 : 0) << '\n';

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
      const auto t_l4_begin = std::chrono::steady_clock::now();
      const auto plan = planner.plan(target, robot_state, timestamp, false);
      ms_l4.push_back(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_l4_begin).count());
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
        entry.made_at = timestamp;
        entry.valid_at = timestamp +
          std::chrono::microseconds(static_cast<long long>(predict_time * 1e6));
        entry.armors = predictor.predict(*target, predict_time).armor_xyza_list();
        entry.armors_hold = target->armor_xyza_list();
        entry.type = L3Estimation::armorTypeOf(target->name)
                       .value_or(L3Estimation::ArmorType::Small);
        entry.name = target->name;
        pending.push_back(std::move(entry));
      }
      while (!pending.empty() && pending.front().valid_at <= timestamp) {
        const PendingPrediction entry = pending.front();
        pending.pop_front();

        // 兑现的这一帧必须真的落在 horizon 附近。sp 的 demo 是十来段素材拼起来
        // 的，接缝处时间戳能跳 170 秒，跨接缝兑现等于拿 A 场景的预测去对 B 场景
        // 的检出，算出来的偏差没有意义。
        const double elapsed =
          std::chrono::duration<double>(timestamp - entry.made_at).count();
        if (elapsed > predict_time + runtime_config.ieskf_tracker.max_frame_gap) {
          continue;
        }

        // 没有检出就没有参照，这条预测作废。
        if (armors.empty()) {
          continue;
        }
        std::vector<cv::Point2f> det_centers;
        for (const auto& detection : armors) {
          det_centers.push_back(detection.center);
        }
        // 用本帧的云台姿态把当时外推的四块板投回像素。云台这段时间转过多少
        // 由这一步吸收，剩下的偏差才是整车运动没预测准的部分。
        const auto reproject = [&](const std::vector<Eigen::Vector4d>& poses) {
          std::vector<std::optional<cv::Point2d>> pixels;
          pixels.reserve(poses.size());
          for (const auto& xyza : poses) {
            pixels.push_back(armorBoxCenter(solver, xyza, entry.type, entry.name));
          }
          return worstAssignment(pixels, det_centers);
        };

        const auto pred_worst = reproject(entry.armors);
        if (!pred_worst) {
          continue;
        }
        const auto hold_worst = reproject(entry.armors_hold);
        pred_pixel_err.push_back(*pred_worst);
        if (hold_worst) {
          hold_pixel_err.push_back(*hold_worst);
        }
        pred_csv << frame_index << ',' << pose.seconds << ',' << predict_time << ','
                 << det_centers.size() << ',' << *pred_worst << ',';
        if (hold_worst) pred_csv << *hold_worst;
        pred_csv << '\n';
      }
    }

    aim_csv.close();
    obs_csv.close();
    frame_csv.close();
    pred_csv.close();

    // 四个物理通道的汇总。bias 取逐帧 mean 的平均，noise 取逐帧 rms 的
    // 平方平均，两者分开看：bias 显著非零说明检测器在那一维有系统偏差，
    // 放大 R 压不住它，得回检测侧修。
    //
    // 注意这里是**创新**不是观测噪声：r = z − h(x̌)，协方差是 S = H·P·Hᵀ + R，
    // 含先验不确定度。所以这几个数能比较各通道的相对权重、能暴露偏差，但
    // 不能直接当 lightCov 的 sigma 用。要标 R 本身得另外量：拿目标基本静止
    // 的录像，把每根灯条的四个端点坐标在短窗口内去趋势，残差的经验协方差
    // 转到灯条系读 sigma 和 rho。
    const auto channel_report = [&]() {
      static constexpr std::array<const char *, 4> kNames{
        "横移⊥ px", "沿移∥ px", "倾角 deg", "长度 px"};
      std::ostringstream out;
      out << std::fixed << std::setprecision(4)
          << "-- 端点创新分四通道（先验点，含先验不确定度，非纯观测噪声）--\n";
      for (std::size_t k = 0; k < kNames.size(); ++k) {
        double square_sum = 0.0;
        for (const double value : res_channel_rms[k]) square_sum += value * value;
        const double noise = res_channel_rms[k].empty()
                               ? 0.0
                               : std::sqrt(square_sum / res_channel_rms[k].size());
        std::vector<double> abs_mean;
        abs_mean.reserve(res_channel_mean[k].size());
        for (const double value : res_channel_mean[k]) abs_mean.push_back(std::abs(value));
        out << kNames[k] << "  bias " << mean(res_channel_mean[k]) << "  noise "
            << noise << "  |bias| p90 " << percentile(abs_mean, 0.9) << '\n';
      }
      return out.str();
    };

    const auto& light_stats = tracker.lightMatchStats();
    std::cout << "\n=== " << input << " ===\n"
              << "帧数                        " << frames << '\n'
              << "有检出的帧                  " << frames_with_det << '\n'
              << "检出总数                    " << det_total << '\n'
              << "Tracking 帧                 " << frames_tracking << '\n'
              << "跟踪丢失/重置次数           " << resets << '\n'
              << "单帧多观测更新的帧          " << double_update_frames << '\n'
              << "侧边灯条 总数/有灯条的帧    " << side_light_total << " / "
              << frames_with_side_light << '\n'
              << "侧边灯条 L3 实际采纳        " << side_light_used << '\n'
              << "  关联跳过帧 无锚板/无候选槽  " << light_stats.frames_skipped
              << " / " << light_stats.frames_no_candidate << '\n'
              << "  候选槽位 " << light_stats.slots << " 个 / 有槽位帧 "
              << (frames - light_stats.frames_skipped
                  - light_stats.frames_no_candidate)
              << "  采纳到灯条的帧 " << light_stats.frames_matched << '\n'
              << "  过门 " << light_stats.passed << " / " << light_stats.considered
              << "  毙于 长度 " << light_stats.reject_length << " / 角度 "
              << light_stats.reject_angle << " / 卡方 "
              << light_stats.reject_chi2 << '\n'
              << "  过门灯条 实测/预测长度 几何平均 "
              << (light_stats.passed > 0
                    ? std::exp(light_stats.log_length_ratio /
                               static_cast<double>(light_stats.passed))
                    : 0.0)
              << '\n'
              << "数字分类采信/丢弃           " << number_accepted << " / "
              << number_dropped << '\n'
              << "角点精修 替换/保留网络      " << refine_hit << " / " << refine_kept
              << "  触发率 "
              << (refine_hit + refine_kept > 0
                    ? 100.0 * static_cast<double>(refine_hit) /
                        static_cast<double>(refine_hit + refine_kept)
                    : 0.0)
              << "%\n"
              << "  其中保留网络的成因  没找到灯条 " << refine_no_bar
              << " / 灯条太短 " << refine_too_short << " / 端点超门限 "
              << refine_shift_rej << '\n'
              << "  轮廓 " << refine_contours << "  过筛 " << refine_bar_kept
              << "  毙于 角度 " << refine_rej_angle << " / 长宽比 "
              << refine_rej_ratio << " / 长度 " << refine_rej_len << '\n'
              << "-- 滤波器 --\n"
              << "NIS  mean " << mean(nis_values) << "  p50 " << percentile(nis_values, 0.5)
              << "  p90 " << percentile(nis_values, 0.9)
              << "  (观测维数每帧在变，见 nis_dof 列，固定卡方门限不适用)\n"
              << "NIS/dof mean " << mean(nis_per_dof) << "  p50 "
              << percentile(nis_per_dof, 0.5) << "  p90 " << percentile(nis_per_dof, 0.9)
              << "  (一致时期望为 1)\n"
              << "v_yaw |mean| " << mean(vyaws) << "  p90 " << percentile(vyaws, 0.9)
              << "  max " << percentile(vyaws, 1.0) << '\n'
              << "r1   mean " << mean(radii) << "  p50 " << percentile(radii, 0.5) << "  max "
              << percentile(radii, 1.0) << '\n'
              << "-- 端点创新（先验点，投到灯条坐标系，px）--\n"
              << "沿灯条     mean " << mean(res_along_stats) << "  p90 "
              << percentile(res_along_stats, 0.9) << "  max "
              << percentile(res_along_stats, 1.0) << '\n'
              << "垂直灯条   mean " << mean(res_perp_stats) << "  p90 "
              << percentile(res_perp_stats, 0.9) << "  max "
              << percentile(res_perp_stats, 1.0) << '\n'
              << channel_report()
              << "-- 单帧耗时（ms）--\n"
              << "L2 检测  p50 " << percentile(ms_l2, 0.5) << "  p90 "
              << percentile(ms_l2, 0.9) << "  max " << percentile(ms_l2, 1.0) << '\n'
              << "  侧边灯条 p50 " << percentile(ms_side_light, 0.5) << "  p90 "
              << percentile(ms_side_light, 0.9) << "  max " << percentile(ms_side_light, 1.0)
              << "  n=" << ms_side_light.size() << '\n'
              << "L3 跟踪  p50 " << percentile(ms_l3, 0.5) << "  p90 "
              << percentile(ms_l3, 0.9) << "  max " << percentile(ms_l3, 1.0) << '\n'
              << "L4 规划  p50 " << percentile(ms_l4, 0.5) << "  p90 "
              << percentile(ms_l4, 0.9) << "  max " << percentile(ms_l4, 1.0) << '\n'
              << "-- 开环预测 " << predict_time * 1e3 << " ms（整车四块板 vs 当帧检出）--\n"
              << "预测   p50 " << percentile(pred_pixel_err, 0.5) << "  p90 "
              << percentile(pred_pixel_err, 0.9) << " px   <=20px "
              << hitRate(pred_pixel_err, 20.0) << "%  <=50px "
              << hitRate(pred_pixel_err, 50.0) << "%  n=" << pred_pixel_err.size() << '\n'
              << "不外推 p50 " << percentile(hold_pixel_err, 0.5) << "  p90 "
              << percentile(hold_pixel_err, 0.9) << " px   <=20px "
              << hitRate(hold_pixel_err, 20.0) << "%  （基准，预测必须比它准）\n"
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
