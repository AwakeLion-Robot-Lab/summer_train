#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace L1Sensor {
class Camera;
}

namespace runtime {

/**
 * @brief 自瞄主程序：装配各层并跑单线程主循环
 */
class AutoAimRuntime {
public:
  /**
   * @brief 只保存路径，不打开任何设备
   * @param config_path 相机配置 YAML 路径（如 config/camera_config.yaml）
   */
  explicit AutoAimRuntime(const std::string& config_path);

  /**
   * @brief 装配并阻塞运行主循环
   *
   * 读 config/auto_aim.yaml、打开相机、加载模型、启动串口，然后逐帧
   * 取图 → 检测 → 跟踪 → 规划 → 开火判定 → 下发命令。stop() 被调用或在叠加层
   * 窗口按 Esc / q 后返回。
   */
  void run();

  /**
   * @brief 请求主循环退出，并停止相机取流以打断阻塞的读帧；可从其他线程调用
   */
  void stop();

private:
  /// 主循环是否继续
  std::atomic<bool> running_{false};

  /// 保护 active_camera_
  mutable std::mutex camera_mutex_;

  /// run() 中正在使用的相机，供 stop() 打断取流
  std::shared_ptr<L1Sensor::Camera> active_camera_;

  /// 相机配置路径
  std::string config_path_;
};

}  // namespace runtime
