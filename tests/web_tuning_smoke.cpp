// WebCmd + applyWebTuning 冒烟：写 cmd JSON，验证白名单热更与忽略未知键。
#include "l6_telemetry/web_cmd.hpp"
#include "runtime/auto_aim_config.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <numbers>

namespace {

bool require(bool condition, const char* message)
{
  if (!condition) {
    std::cerr << "web_tuning_smoke: FAIL " << message << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main()
{
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;

  runtime::AutoAimConfig config;
  const double default_q = config.target.q_translation;
  const bool default_overlay = config.debug.overlay;
  const std::string default_mode = config.debug.force_work_mode;

  const nlohmann::json command = {
    {"plan", {{"send_to_control_ms", 12.5}}},
    {"target", {{"q_translation", 250.0}}},
    {"fire", {{"hit_margin_ratio", 0.8}, {"min_yaw_tolerance_deg", 1.5}}},
    {"debug", {{"force_work_mode", "auto_aim"}, {"overlay", true}}},
    {"unknown_group", {{"whatever", 1.0}}},
    {"fire", {{"shoot_enable", true}}},  // 故意不在白名单
  };
  // 上面重复的 fire 键会合并；显式补一个 shoot_enable 保证被忽略。
  nlohmann::json merged = command;
  merged["fire"]["shoot_enable"] = true;

  const auto changed = runtime::applyWebTuning(config, merged);

  if (!require(config.plan.impact.send_to_control.has_value(), "send_to_control missing")) {
    return 1;
  }
  if (!require(
        std::abs(*config.plan.impact.send_to_control - 0.0125) < 1e-9,
        "send_to_control ms->s")) {
    return 2;
  }
  if (!require(config.target.q_translation == 250.0, "q_translation not applied")) {
    return 3;
  }
  if (!require(std::abs(config.fire.hit_margin_ratio - 0.8) < 1e-9, "hit_margin")) {
    return 4;
  }
  if (!require(
        std::abs(config.fire.min_yaw_tolerance * kRadToDeg - 1.5) < 1e-9,
        "min_yaw_tolerance deg->rad")) {
    return 5;
  }
  if (!require(config.debug.overlay != default_overlay, "overlay not applied")) {
    return 6;
  }
  if (!require(config.debug.force_work_mode == "auto_aim", "force_work_mode")) {
    return 7;
  }
  if (!require(config.fire.shoot_enable == false, "shoot_enable must stay false")) {
    return 8;
  }
  if (!require(!changed.empty(), "changed list empty")) {
    return 9;
  }
  const auto unchanged = runtime::applyWebTuning(config, merged);
  if (!require(unchanged.empty(), "same command must be idempotent")) {
    return 10;
  }
  (void)default_q;
  (void)default_mode;

  // WebCmd 文件通道
  const char* path = "/dev/shm/newvision_cmd.json";
  {
    std::ofstream out(path);
    out << R"({"plan":{"send_to_control_ms":9.0}})";
  }
  L6Telemetry::WebCmd cmd;
  if (!require(cmd.poll(), "poll failed")) {
    return 11;
  }
  if (!require(cmd.has("plan.send_to_control_ms"), "has path")) {
    return 12;
  }
  const auto value = cmd.value("plan.send_to_control_ms");
  if (!require(value.is_number() && value.get<double>() == 9.0, "value")) {
    return 13;
  }

  std::printf("web_tuning_smoke: ok changed=%zu\n", changed.size());
  return 0;
}
