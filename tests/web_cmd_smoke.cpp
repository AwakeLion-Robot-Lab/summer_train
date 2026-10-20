// WebCmd 反向通道冒烟：写 /dev/shm/newvision_cmd.json，验证路径查找与缓存。
#include "l6_telemetry/web_cmd.hpp"

#include <fstream>
#include <iostream>

namespace {

bool require(bool condition, const char* message)
{
  if (!condition) {
    std::cerr << "web_cmd_smoke: FAIL " << message << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main()
{
  const char* path = "/dev/shm/newvision_cmd.json";
  {
    std::ofstream out(path);
    out << R"({"plan":{"send_to_control_ms":9.0},"debug":{"force_work_mode":"auto_aim"}})";
  }

  L6Telemetry::WebCmd cmd;
  if (!require(cmd.poll(), "poll failed")) {
    return 1;
  }
  if (!require(cmd.has("plan.send_to_control_ms"), "nested path")) {
    return 2;
  }
  if (!require(cmd.has("debug.force_work_mode"), "string path")) {
    return 3;
  }
  const auto value = cmd.value("plan.send_to_control_ms");
  if (!require(value.is_number() && value.get<double>() == 9.0, "number value")) {
    return 4;
  }
  const auto mode = cmd.value("debug.force_work_mode");
  if (!require(mode.is_string() && mode.get<std::string>() == "auto_aim", "string value")) {
    return 5;
  }
  if (!require(!cmd.has("nope.missing"), "missing path")) {
    return 6;
  }

  // lookupPath 直接测扁平键
  const nlohmann::json flat = {{"a.b", 1.5}};
  const auto flat_value = L6Telemetry::lookupPath(flat, "a.b");
  if (!require(flat_value.has_value() && flat_value->get<double>() == 1.5, "flat key")) {
    return 7;
  }

  std::printf("web_cmd_smoke: ok\n");
  return 0;
}
