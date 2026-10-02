#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/robot_state.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"
#include "l5_control/serial_command.hpp"
#include "l6_telemetry/udp_json_sender.hpp"
#include "runtime/auto_aim_config.hpp"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <optional>

namespace runtime {

// 一帧要交给调试旁路的东西，全部是主循环里已有的量。
struct DebugFrame {
  std::chrono::steady_clock::time_point timestamp;
  // Idle 不规划，下面三项为空。
  std::optional<std::chrono::steady_clock::time_point> plan_time;
  std::optional<Eigen::Quaterniond> image_pose;
  std::optional<Eigen::Quaterniond> actual_pose;
  bool aiming{false};
  bool pose_ready{false};
  double pose_wait_ms{0.0};
  const L1Sensor::RobotState* state{nullptr};
  const L1Sensor::SerialWorker* serial{nullptr};
  const L2Perception::ArmorFrame* perception{nullptr};
  const L3Estimation::EskfTracker* tracker{nullptr};
  const std::optional<L3Estimation::EskfTarget>* target{nullptr};
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};
  const L4Planning::Plan* plan{nullptr};
  const L5Control::FireDecision* fire{nullptr};
  const std::optional<L5Control::SerialCommand>* command{nullptr};
};

// 自瞄主循环的调试旁路：叠加层窗口、PlotJuggler 遥测、强制 WorkMode。
// 只服务调试，不参与瞄准链路；debug: 全关时每帧只剩几个判断。
class AutoAimDebug {
public:
  // 启动时调用一次：按 debug: 开窗口、建遥测、解析 force_work_mode。
  AutoAimDebug(
    const AutoAimConfig& config,
    const std::optional<L1Sensor::CameraCalibration>& calibration);
  ~AutoAimDebug();

  AutoAimDebug(const AutoAimDebug&) = delete;
  AutoAimDebug& operator=(const AutoAimDebug&) = delete;

  // debug.force_work_mode 解析结果；空表示不覆盖下位机的 WorkMode。
  std::optional<L1Sensor::WorkMode> forcedMode() const noexcept
  {
    return forced_mode_;
  }

  // 命令下发之后调用：发遥测，按 overlay_every 在 image 上画叠加层。
  void record(cv::Mat& image, const DebugFrame& in);

  // 每轮循环末尾调用：数帧率，把它写在 image 上并刷新叠加层窗口。
  // 按了 Esc / q 返回 false。
  [[nodiscard]] bool show(cv::Mat& image);

private:
  bool overlay_{false};
  int overlay_every_{1};
  std::uint64_t frame_index_{0};
  // 主循环帧率：每满 1 s 结算一次窗口内 show() 的调用次数。读帧失败的轮次
  // 不调 show()，所以这是真正处理过的帧数，不是相机出帧率。
  std::chrono::steady_clock::time_point fps_start_;
  int fps_frames_{0};
  double fps_{0.0};
  std::optional<L1Sensor::CameraCalibration> calibration_;
  // 叠加层要把世界系位姿投回图像，需要一个求解器。这里另建一个与 Tracker
  // 内部同参数的实例，只做重投影、不参与滤波——不为了画图去开 Tracker 的内部。
  std::optional<L3Estimation::PnpSolver> solver_;
  std::optional<L6Telemetry::UdpJsonSender> plotter_;
  std::optional<L1Sensor::WorkMode> forced_mode_;
};

}  // namespace runtime
