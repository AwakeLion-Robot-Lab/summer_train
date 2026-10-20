#include "runtime/runtime_io.hpp"

#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l1_sensor/talos/gimbal_codec.hpp"
#include "l1_sensor/talos/talos_config.hpp"
#include "l1_sensor/talos/talos_reader.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <cmath>
#include <numbers>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>
#include <yaml-cpp/yaml.h>

namespace runtime {
namespace {

class HardwareIo final : public RuntimeIo {
public:
  explicit HardwareIo(const std::string& camera_config)
    : camera_(std::make_shared<L1Sensor::Camera>(camera_config)),
      serial_(L1Sensor::loadSerialConfig("config/serial_config.yaml")) {}

  bool start() override { return serial_.start(); }
  bool read(cv::Mat& image, std::chrono::steady_clock::time_point& timestamp) override
  {
    if (!camera_->read(image, timestamp)) return false;
    ++received_frames_;
    return true;
  }
  const std::optional<L1Sensor::CameraCalibration>& calibration() const noexcept override
  { return camera_->calibration(); }
  std::optional<L1Sensor::RobotState> latestState() const override { return serial_.latestState(); }
  bool waitPose(std::chrono::steady_clock::time_point timestamp) const override
  { return serial_.waitPose(timestamp); }
  std::optional<Eigen::Quaterniond> gimbalPoseAt(
    std::chrono::steady_clock::time_point timestamp) const override
  { return serial_.gimbalPoseAt(timestamp); }
  std::optional<Eigen::Quaterniond> latestGimbalPose() const override
  { return serial_.latestGimbalPose(); }
  void updateCommand(const L5Control::SerialCommand& command) override { serial_.updateCommand(command); }
  RuntimeIoStats stats() const override
  {
    const auto camera = camera_->stats();
    return {serial_.receivedStateCount(), serial_.droppedPacketCount(),
            serial_.skippedByteCount(), serial_.sentCommandCount(),
            serial_.failedCommandCount(), serial_.poseBeforeHistoryCount(),
            serial_.poseAfterHistoryCount(), camera.captured_frames,
            camera.dropped_frames, received_frames_};
  }
  std::optional<L1Sensor::talos::GroundTruthBatch> groundTruth() const override
  { return std::nullopt; }
  std::string name() const override { return "hardware"; }
  void stop() override { serial_.stop(); camera_->stop(); }

  bool setExposureGain(double exposure_ms, double gain) override
  {
    return camera_ && camera_->setExposureGain(exposure_ms, gain);
  }

private:
  std::shared_ptr<L1Sensor::Camera> camera_;
  L1Sensor::SerialWorker serial_;
  std::uint64_t received_frames_{};
};

L1Sensor::CameraCalibration calibrationFromTalos(const L1Sensor::talos::CameraInfo& info)
{
  L1Sensor::CameraCalibration calibration;
  calibration.image_size = {static_cast<int>(info.width), static_cast<int>(info.height)};
  calibration.camera_matrix = cv::Mat::zeros(3, 3, CV_64F);
  calibration.camera_matrix.at<double>(0, 0) = info.fx;
  calibration.camera_matrix.at<double>(1, 1) = info.fy;
  calibration.camera_matrix.at<double>(0, 2) = info.cx;
  calibration.camera_matrix.at<double>(1, 2) = info.cy;
  calibration.camera_matrix.at<double>(2, 2) = 1.0;
  calibration.distortion_coefficients = cv::Mat::zeros(5, 1, CV_64F);
  for (int index = 0; index < 5; ++index) {
    calibration.distortion_coefficients.at<double>(index) = info.distortion[index];
  }
  // 图像仍是 OpenCV 光学系；Talos 姿态则已由仿真器发布为 world<-barrel。
  Eigen::Isometry3d optical_to_barrel = Eigen::Isometry3d::Identity();
  optical_to_barrel.linear() << 0.0, 0.0, 1.0,
                                -1.0, 0.0, 0.0,
                                 0.0, -1.0, 0.0;
  calibration.T_barrel_camera = optical_to_barrel;
  return calibration;
}

class TalosIo final : public RuntimeIo {
public:
  explicit TalosIo(const std::string& path)
    : config_(L1Sensor::talos::loadTalosConfig(path)),
      reader_(std::make_shared<L1Sensor::talos::TalosReader>(config_.shm_dir))
  {
    if (!reader_->open()) {
      throw std::runtime_error(
        "Talos shared memory unavailable in " + config_.shm_dir +
        "; start bevy_robomaster_simulator first");
    }
    calibration_ = config_.calibration
      ? config_.calibration
      : std::optional<L1Sensor::CameraCalibration>{calibrationFromTalos(reader_->cameraInfo())};
  }

  bool start() override { return reader_->isOpen(); }

  bool read(cv::Mat& image, std::chrono::steady_clock::time_point& timestamp) override
  {
    L1Sensor::talos::TalosFrame frame;
    if (!reader_->readFrame(frame, config_.timeout) || !frame.rgb) {
      ++consecutive_timeouts_;
      // 仿真器重启会 unlink 并重建两个共享内存文件；旧 mmap 仍合法但永远
      // 不会再更新。连续超时后重新打开路径，恢复时仍由同一个 Reader 消费四路。
      if (consecutive_timeouts_ >= 10) {
        reader_->reopen();
        hold_initialized_ = false;
        consecutive_timeouts_ = 0;
      }
      return false;
    }
    consecutive_timeouts_ = 0;
    if (!hold_initialized_) {
      if (const auto pose = reader_->pose(L1Sensor::talos::PoseIndex::Gimbal)) {
        if (const auto quaternion = poseQuaternion(*pose)) {
          const Eigen::Vector3d rpy =
            L6Telemetry::rotationToRpy(quaternion->toRotationMatrix());
          // 在任何算法命令之前覆盖共享内存里的旧命令。推荐启动顺序是先视觉、
          // 后按 F5；这样 F5 首次消费的一定是当前世界姿态的禁火保持。
          const L5Control::SerialCommand hold{rpy.z(), rpy.y(), false};
          reader_->writeGimbalCmd(L1Sensor::talos::encodeGimbalCommand(
            hold, config_.default_distance_m, false));
          ++sent_commands_;
          hold_initialized_ = true;
        }
      }
    }
    const cv::Mat rgb(static_cast<int>(frame.height), static_cast<int>(frame.width),
                      CV_8UC3, const_cast<std::uint8_t*>(frame.rgb));
    cv::cvtColor(rgb, image, cv::COLOR_RGB2BGR);
    timestamp = reader_->toSteadyTime(frame.timestamp_ns);
    ++received_frames_;
    return true;
  }

  const std::optional<L1Sensor::CameraCalibration>& calibration() const noexcept override
  { return calibration_; }

  std::optional<L1Sensor::RobotState> latestState() const override
  {
    const auto pose = reader_->pose(L1Sensor::talos::PoseIndex::Gimbal);
    if (!pose) return std::nullopt;
    const auto quaternion = poseQuaternion(*pose);
    if (!quaternion) return std::nullopt;
    const Eigen::Vector3d rpy = L6Telemetry::rotationToRpy(quaternion->toRotationMatrix());
    L1Sensor::RobotState state;
    state.rpy.roll = rpy.x();
    // Talos 发布的 world<-barrel 已按项目右手绕 Y 约定：正值低头，负值抬头。
    state.rpy.pitch = rpy.y();
    state.rpy.yaw = rpy.z();
    state.bullet_speed = config_.bullet_speed;
    state.enemy_color = config_.enemy_color;
    state.mode = config_.mode;
    state.timestamp = reader_->toSteadyTime(pose->timestamp_ns);
    return state;
  }

  bool waitPose(std::chrono::steady_clock::time_point) const override
  { return reader_->pose(L1Sensor::talos::PoseIndex::Gimbal).has_value(); }

  std::optional<Eigen::Quaterniond> gimbalPoseAt(
    std::chrono::steady_clock::time_point) const override
  {
    const auto pose = reader_->pose(L1Sensor::talos::PoseIndex::Gimbal);
    return pose ? poseQuaternion(*pose) : std::nullopt;
  }

  std::optional<Eigen::Quaterniond> latestGimbalPose() const override
  { return gimbalPoseAt({}); }

  void updateCommand(const L5Control::SerialCommand& command) override
  {
    if (const auto pose = reader_->pose(L1Sensor::talos::PoseIndex::Gimbal)) {
      if (const auto quaternion = poseQuaternion(*pose)) {
        const Eigen::Vector3d rpy =
          L6Telemetry::rotationToRpy(quaternion->toRotationMatrix());
        const double yaw_delta = L6Telemetry::limit_rad(command.yaw - rpy.z());
        const double actual_pitch = rpy.y();
        const double pitch_delta = command.pitch - actual_pitch;
        constexpr double kRadToDeg = 180.0 / std::numbers::pi;
        if (!std::isfinite(yaw_delta) || !std::isfinite(pitch_delta) ||
            std::abs(yaw_delta) * kRadToDeg > config_.max_yaw_jump_deg ||
            std::abs(pitch_delta) * kRadToDeg > config_.max_pitch_jump_deg) {
          // Talos 专用安全门，不改变 L2-L5：面前目标不可能在一帧内要求几十度
          // 跳转。拒绝异常命令并持续覆盖为当前位置，确保 F5 不会消费危险角度。
          const L5Control::SerialCommand hold{rpy.z(), actual_pitch, false};
          reader_->writeGimbalCmd(L1Sensor::talos::encodeGimbalCommand(
            hold, config_.default_distance_m, false));
          ++sent_commands_;
          ++failed_commands_;
          if (!jump_warning_active_) {
            L6Telemetry::logWarn(
              "Talos command jump blocked; actual_yaw/pitch desired_yaw/pitch delta_yaw/pitch(deg)",
              rpy.z() * kRadToDeg, actual_pitch * kRadToDeg,
              command.yaw * kRadToDeg, command.pitch * kRadToDeg,
              yaw_delta * kRadToDeg, pitch_delta * kRadToDeg);
          }
          jump_warning_active_ = true;
          return;
        }
      }
    }
    jump_warning_active_ = false;
    // -90° 水平的偏置属于 Talos 协议编码，不是算法坐标系转换。
    const auto output = L1Sensor::talos::encodeGimbalCommand(
      command, config_.default_distance_m, config_.shoot_enable);
    reader_->writeGimbalCmd(output);
    ++sent_commands_;
  }

  RuntimeIoStats stats() const override
  {
    return {received_frames_, 0, 0, sent_commands_, failed_commands_, 0, 0,
            received_frames_, 0, received_frames_};
  }
  std::optional<L1Sensor::talos::GroundTruthBatch> groundTruth() const override
  {
    const auto value = reader_->groundTruth();
    return value.timestamp_ns == 0 ? std::nullopt
                                   : std::optional<L1Sensor::talos::GroundTruthBatch>{value};
  }
  std::string name() const override { return "talos"; }
  void stop() override {}

private:
  static std::optional<Eigen::Quaterniond> poseQuaternion(const L1Sensor::talos::PoseMeta& pose)
  {
    Eigen::Quaterniond value(pose.quaternion[0], pose.quaternion[1],
                             pose.quaternion[2], pose.quaternion[3]);
    if (!value.coeffs().allFinite() || value.norm() < 1e-9) return std::nullopt;
    // 当前仿真器工作区已在 capture.rs 完成 Bevy -> 项目 barrel 转换。
    // 这里直接使用，禁止再次乘固定轴矩阵造成二次转换。
    return value.normalized();
  }

  L1Sensor::talos::TalosConfig config_;
  std::shared_ptr<L1Sensor::talos::TalosReader> reader_;
  std::optional<L1Sensor::CameraCalibration> calibration_;
  std::uint64_t received_frames_{};
  std::uint64_t sent_commands_{};
  std::uint64_t failed_commands_{};
  int consecutive_timeouts_{};
  bool hold_initialized_{};
  bool jump_warning_active_{};
};

} // namespace

std::shared_ptr<RuntimeIo> makeRuntimeIo(const std::string& camera_config_path)
{
  const YAML::Node root = YAML::LoadFile(camera_config_path);
  const std::string camera_name = root["camera_name"]
    ? root["camera_name"].as<std::string>() : std::string{};
  if (camera_name == "talos") return std::make_shared<TalosIo>(camera_config_path);
  return std::make_shared<HardwareIo>(camera_config_path);
}

} // namespace runtime
