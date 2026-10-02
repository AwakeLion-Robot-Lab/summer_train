#include "runtime/auto_aim_debug.hpp"
#include "l6_telemetry/aim_overlay.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include <opencv2/highgui.hpp>

#include <format>
#include <numbers>
#include <string>

namespace {

// 把配置里的字符串解析成 WorkMode。空串或无法识别都返回空，表示不覆盖——
// 拼错模式名不该悄悄退化成 Idle 或 AutoAim，那两种误判的后果完全不同。
std::optional<L1Sensor::WorkMode> parseWorkMode(const std::string& name)
{
  if (name.empty()) {
    return std::nullopt;
  }
  if (name == "auto_aim") return L1Sensor::WorkMode::AutoAim;
  if (name == "outpost") return L1Sensor::WorkMode::Outpost;
  if (name == "small_buff") return L1Sensor::WorkMode::SmallBuff;
  if (name == "big_buff") return L1Sensor::WorkMode::BigBuff;
  if (name == "idle") return L1Sensor::WorkMode::Idle;
  LOG_ERROR(
    "debug.force_work_mode is not a known mode, ignored:", name,
    "| valid: auto_aim outpost small_buff big_buff idle");
  return std::nullopt;
}

// PlotJuggler 遥测。分节点是硬要求：曝光时刻云台角 gimbal/、规划时刻云台角
// gimbal_now/、真正下发的命令 cmd/ 必须落在不同分支上，三者同图才看得出
// 跟随误差和振荡；观测计数 track/ 与滤波状态 ekf/ 同理。角度 degree、时间 ms。
nlohmann::json telemetryFrame(const runtime::DebugFrame& in, double fps)
{
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  const auto ms = [](auto duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  };
  const auto putYawPitch = [&](nlohmann::json& node, const Eigen::Quaterniond& q) {
    const Eigen::Vector3d ypr = L6Telemetry::eulers(q.toRotationMatrix(), 2, 1, 0);
    node["yaw"] = ypr[0] * kRadToDeg;
    node["pitch"] = ypr[1] * kRadToDeg;
  };
  nlohmann::json data;

  // 帧曝光时刻，单位 s。必须发：主循环周期本身在抖，按 UDP 到达时刻排点会把
  // 平滑曲线画成台阶。在 PlotJuggler 的 UDP/JSON 插件里选它作 timestamp。
  data["t"] = std::chrono::duration<double>(in.timestamp.time_since_epoch()).count();
  data["aiming"] = in.aiming ? 1 : 0;
  data["fps"] = fps;

  if (in.image_pose) {
    putYawPitch(data["gimbal"], *in.image_pose);
  }
  if (in.actual_pose) {
    putYawPitch(data["gimbal_now"], *in.actual_pose);
  }
  data["gimbal"]["bullet_speed"] = in.state->bullet_speed;

  // cmd: 本帧 L5 交给串口的命令（含规划失败时的 safeHold），振荡看这条。
  if (*in.command) {
    data["cmd"]["yaw"] = (*in.command)->yaw * kRadToDeg;
    data["cmd"]["pitch"] = (*in.command)->pitch * kRadToDeg;
    data["cmd"]["shoot"] = (*in.command)->shoot ? 1 : 0;
  }
  data["cmd"]["sent"] = in.command->has_value() ? 1 : 0;

  // serial: img_minus_imu 是图像曝光时刻减最新一帧姿态的接收时刻。为正说明
  // 图像比最新姿态还新、gimbalPoseAt 只能取边界值不做插值；呈锯齿说明串口
  // 批量收包，图像与姿态实际没对齐。云台一动，这个误差就变成假的目标运动。
  data["serial"]["img_minus_imu"] = ms(in.timestamp - in.state->timestamp);
  if (in.plan_time) {
    data["serial"]["plan_minus_imu"] = ms(*in.plan_time - in.state->timestamp);
  }
  // pose_wait 是检测前等姿态的时间；pose_ready=0 说明等满了也没等到。
  data["serial"]["pose_ready"] = in.pose_ready ? 1 : 0;
  data["serial"]["pose_wait"] = in.pose_wait_ms;
  data["serial"]["rx"] = in.serial->receivedStateCount();
  data["serial"]["rx_dropped"] = in.serial->droppedPacketCount();
  data["serial"]["tx"] = in.serial->sentCommandCount();
  data["serial"]["tx_failed"] = in.serial->failedCommandCount();

  // track: 状态机与本帧观测量。drops 是单调计数，单帧重建在状态图上看不见，
  // 在它上面一定留下台阶。
  data["track"]["state"] = static_cast<int>(in.track_state);
  data["track"]["n_armors"] = static_cast<int>(in.perception->armors.size());
  data["track"]["n_lights"] = static_cast<int>(in.perception->lights.size());
  if (in.tracker) {
    data["track"]["n_used_lights"] = static_cast<int>(in.tracker->usedLights().size());
    data["track"]["match_count"] = in.tracker->lastMatchCount();
    data["track"]["drops"] = static_cast<std::uint64_t>(in.tracker->dropCount());
  }

  // ekf: 整车 13 维，按 VehicleModel::idx 的顺序命名。ekf_x() 吐的是线性半径。
  if (*in.target) {
    namespace idx = L3Estimation::VehicleModel::idx;
    const auto& target = **in.target;
    const Eigen::VectorXd x = target.ekf_x();
    auto& ekf = data["ekf"];
    ekf["x"] = x[idx::CX];
    ekf["vx"] = x[idx::VCX];
    ekf["y"] = x[idx::CY];
    ekf["vy"] = x[idx::VCY];
    ekf["z"] = x[idx::CZ];
    ekf["vz"] = x[idx::VCZ];
    ekf["yaw"] = x[idx::ROT_Z] * kRadToDeg;
    ekf["w"] = x[idx::VYAW];
    ekf["r1"] = x[idx::LOG_R1];
    ekf["p1"] = x[idx::P1];
    ekf["p2"] = x[idx::P2];
    ekf["tilt_pitch"] = x[idx::ROT_Y] * kRadToDeg;
    ekf["tilt_roll"] = x[idx::ROT_X] * kRadToDeg;
    ekf["last_id"] = target.last_id;
    ekf["jumped"] = target.jumped ? 1 : 0;
    ekf["nis"] = target.lastNis();
    ekf["nis_dof"] = target.lastNisDof();
  }

  // aim: L4 规划出的云台目标角，和 cmd/ 的差别只在规划失败的帧。
  // error 是 PlanError：0 成功，1 无目标，2 窗口外，3 弹道无解。
  const L4Planning::Plan& plan = *in.plan;
  data["aim"]["error"] = static_cast<int>(plan.error);
  if (plan.valid()) {
    data["aim"]["yaw"] = plan.aim.yaw * kRadToDeg;
    data["aim"]["pitch"] = plan.aim.pitch * kRadToDeg;
  }
  data["aim"]["armor_id"] = plan.fire ? plan.fire->armor_id : -1;

  // delay: 五段延迟链分开发，绝不合并成一个标量。
  const L4Planning::Delay& delay = plan.timing.delay;
  data["delay"]["image_to_plan"] = delay.image_to_plan * 1e3;
  data["delay"]["plan_to_send"] = delay.plan_to_send * 1e3;
  data["delay"]["send_to_control"] = delay.send_to_control * 1e3;
  data["delay"]["control_to_fire"] = delay.control_to_fire * 1e3;
  data["delay"]["fire_to_hit"] = delay.fire_to_hit * 1e3;
  data["delay"]["before_fire"] = delay.beforeFire() * 1e3;

  // reason 是第一条 RejectReason 的枚举值，-1 表示没有拒绝。
  const L5Control::FireDecision& fire = *in.fire;
  data["fire"]["feasible"] = fire.fire_feasible ? 1 : 0;
  data["fire"]["shoot"] = fire.shoot ? 1 : 0;
  data["fire"]["yaw_err"] = fire.yaw_error * kRadToDeg;
  data["fire"]["pitch_err"] = fire.pitch_error * kRadToDeg;
  data["fire"]["tol_yaw"] = fire.tolerance.yaw * kRadToDeg;
  data["fire"]["tol_pitch"] = fire.tolerance.pitch * kRadToDeg;
  data["fire"]["reason"] =
    fire.reasons.empty() ? -1 : static_cast<int>(fire.reasons.front());

  return data;
}

}  // namespace

namespace runtime {

AutoAimDebug::AutoAimDebug(
  const AutoAimConfig& config,
  const std::optional<L1Sensor::CameraCalibration>& calibration)
    : overlay_(config.debug.overlay),
      overlay_every_(config.debug.overlay_every),
      fps_start_(std::chrono::steady_clock::now()),
      calibration_(calibration)
{
  // 叠加层默认关闭：imshow 的耗时会计进 image_to_plan，而且比赛用的机器
  // 没有显示器，无条件 namedWindow 会直接抛。
  if (overlay_) {
    cv::namedWindow("auto_aim", cv::WINDOW_NORMAL);
    if (calibration_) {
      solver_.emplace(*calibration_, config.armor);
    }
  }

  // 曲线遥测与叠加层各自独立开关：实车上没有显示器，要的恰好是曲线。
  // UDP 是无连接的，没人接收也不会阻塞或报错。
  if (config.debug.plot) {
    plotter_.emplace(
      config.debug.plot_host,
      static_cast<std::uint16_t>(config.debug.plot_port));
    LOG_INFO("telemetry enabled", plotter_->host(), plotter_->port());
  }

  // 调试旁路：强制 WorkMode。启动时解析一次，循环里只做覆盖。
  forced_mode_ = parseWorkMode(config.debug.force_work_mode);
  if (forced_mode_) {
    LOG_WARN(
      "!!! debug.force_work_mode is ACTIVE:", L1Sensor::toString(*forced_mode_),
      "- the MCU's WorkMode is being IGNORED. Clear this key before a match.");
  }
}

AutoAimDebug::~AutoAimDebug()
{
  if (overlay_) {
    cv::destroyWindow("auto_aim");
  }
}

void AutoAimDebug::record(cv::Mat& image, const DebugFrame& in)
{
  if (plotter_) {
    (void)plotter_->send(telemetryFrame(in, fps_));
  }
  if (solver_ && in.tracker && frame_index_ % overlay_every_ == 0) {
    solver_->set_R_world_barrel(in.image_pose);
    L6Telemetry::drawAimOverlay(
      image,
      {.detections = in.perception->armors,
       .target = *in.target,
       .track_state = in.track_state,
       .plan = *in.plan,
       .fire = *in.fire,
       .q_world_barrel = in.image_pose},
      *solver_, *calibration_);
  }
}

bool AutoAimDebug::show(cv::Mat& image)
{
  ++frame_index_;
  ++fps_frames_;
  const auto now = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(now - fps_start_).count();
  if (elapsed >= 1.0) {
    fps_ = fps_frames_ / elapsed;
    fps_frames_ = 0;
    fps_start_ = now;
  }

  if (!overlay_) {
    return true;
  }
  // 状态行在 y=28，帧率写在它下面一行。
  L6Telemetry::drawOutlinedText(
    image, std::format("fps {:.1f}", fps_), {12, 56}, {255, 255, 255}, 0.7);
  cv::imshow("auto_aim", image);
  const int key = cv::waitKey(1);
  return key != 27 && key != 'q' && key != 'Q';
}

}  // namespace runtime
