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

/**
 * @brief 一帧要交给调试旁路的东西，全部是主循环里已有的量
 *
 * 指针都不持有，只在 record() 调用期间有效。
 */
struct DebugFrame {
  /// 图像时间戳
  std::chrono::steady_clock::time_point timestamp;

  /// 规划开始时刻。Idle 不规划，这一项和下面两项为空。
  std::optional<std::chrono::steady_clock::time_point> plan_time;

  /// 曝光时刻的枪管姿态
  std::optional<Eigen::Quaterniond> image_pose;

  /// 规划时刻的枪管实际姿态
  std::optional<Eigen::Quaterniond> actual_pose;

  /// 本帧是否在自瞄
  bool aiming{false};

  /// waitPose 是否等到了曝光之后的那包姿态
  bool pose_ready{false};

  /// waitPose 实际等待的时长，ms
  double pose_wait_ms{0.0};

  /// 下位机状态
  const L1Sensor::RobotState* state{nullptr};

  /// 串口收发线程，读计数器用
  const L1Sensor::SerialWorker* serial{nullptr};

  /// L2 感知结果
  const L2Perception::ArmorFrame* perception{nullptr};

  /// L3 跟踪器；未构造时为 nullptr
  const L3Estimation::EskfTracker* tracker{nullptr};

  /// L3 目标快照
  const std::optional<L3Estimation::EskfTarget>* target{nullptr};

  /// 跟踪状态
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};

  /// L4 规划
  const L4Planning::Plan* plan{nullptr};

  /// L5 开火判定
  const L5Control::FireDecision* fire{nullptr};

  /// 本帧下发的命令
  const std::optional<L5Control::SerialCommand>* command{nullptr};
};

/**
 * @brief 自瞄主循环的调试旁路：叠加层窗口、PlotJuggler 遥测、强制 WorkMode
 *
 * 只服务调试，不参与瞄准链路；debug: 全关时每帧只剩几个判断。
 */
class AutoAimDebug {
public:
  /**
   * @brief 启动时调用一次：按 debug: 开窗口、建遥测、解析 force_work_mode
   * @param config 自瞄配置，读其中的 debug: 与 armor:
   * @param calibration 相机标定；为空时叠加层不画世界系内容
   */
  AutoAimDebug(
    const AutoAimConfig& config,
    const std::optional<L1Sensor::CameraCalibration>& calibration);

  /**
   * @brief 关闭叠加层窗口
   */
  ~AutoAimDebug();

  AutoAimDebug(const AutoAimDebug&) = delete;
  AutoAimDebug& operator=(const AutoAimDebug&) = delete;

  /**
   * @brief debug.force_work_mode 的解析结果
   * @return 强制的模式；空表示不覆盖下位机的 WorkMode
   */
  std::optional<L1Sensor::WorkMode> forcedMode() const noexcept
  {
    return forced_mode_;
  }

  /**
   * @brief 命令下发之后调用：发遥测，按 overlay_every 在 image 上画叠加层
   * @param image 当前帧，叠加层就地画在上面
   * @param in 本帧各层的结果
   */
  void record(cv::Mat& image, const DebugFrame& in);

  /**
   * @brief 每轮循环末尾调用：数帧率，把它写在 image 上并刷新叠加层窗口
   * @param image 当前帧
   * @return 按了 Esc / q 返回 false
   */
  [[nodiscard]] bool show(cv::Mat& image);

private:
  /// 是否开叠加层
  bool overlay_{false};

  /// 每 N 帧画一次叠加层
  int overlay_every_{1};

  /// 已处理的帧数
  std::uint64_t frame_index_{0};

  /// 帧率统计窗口的起点。主循环帧率：每满 1 s 结算一次窗口内 show() 的调用
  /// 次数。读帧失败的轮次不调 show()，所以这是真正处理过的帧数，不是相机出帧率。
  std::chrono::steady_clock::time_point fps_start_;

  /// 当前窗口内的帧数
  int fps_frames_{0};

  /// 上一个窗口结算出的帧率
  double fps_{0.0};

  /// 相机标定
  std::optional<L1Sensor::CameraCalibration> calibration_;

  /// 叠加层要把世界系位姿投回图像，需要一个求解器。这里另建一个与 Tracker
  /// 内部同参数的实例，只做重投影、不参与滤波——不为了画图去开 Tracker 的内部。
  std::optional<L3Estimation::PnpSolver> solver_;

  /// PlotJuggler 遥测，debug.plot 关闭时为空
  std::optional<L6Telemetry::UdpJsonSender> plotter_;

  /// 强制的工作模式
  std::optional<L1Sensor::WorkMode> forced_mode_;
};

}  // namespace runtime
