#pragma once

#include <nlohmann/json.hpp>

#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace L6Telemetry {

// 在线调参的反向通道：web.py 把滑条值写进 /dev/shm/newvision_cmd.json，
// runtime 每 N 帧读一次并按白名单热更新。与 WebSink 对称，只进不出。
//
// 命令格式是嵌套 JSON，例如：
//   {"plan": {"send_to_control_ms": 8.0},
//    "fire": {"hit_margin_ratio": 0.6},
//    "debug": {"force_work_mode": "auto_aim"}}
// 只接受白名单路径；未知键忽略，不报错（前端可能写旧键）。
class WebCmd {
public:
  explicit WebCmd(std::string path = kPath) : path_(std::move(path)) {}
  // 读文件。内容与上次不同才更新缓存；文件不存在返回 false（尚未下发）。
  bool poll();
  [[nodiscard]] bool has(const std::string& dotted_path) const;
  [[nodiscard]] nlohmann::json value(const std::string& dotted_path) const;
  // 整棵命令树，供 apply 层遍历白名单。
  [[nodiscard]] nlohmann::json snapshot() const;

  static constexpr const char* kPath = "/dev/shm/newvision_cmd.json";

private:
  std::string path_;
  mutable std::mutex mutex_;
  nlohmann::json command_ = nlohmann::json::object();
  std::string raw_signature_;
};

// 嵌套路径 "plan.send_to_control_ms" 取值；不存在返回 nullopt。
[[nodiscard]] std::optional<nlohmann::json> lookupPath(
  const nlohmann::json& root, const std::string& dotted_path);

}  // namespace L6Telemetry
