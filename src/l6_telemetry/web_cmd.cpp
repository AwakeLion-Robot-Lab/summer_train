#include "l6_telemetry/web_cmd.hpp"

#include <fstream>
#include <sstream>

namespace L6Telemetry {

namespace {

bool readText(const char* path, std::string& out)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  out = buffer.str();
  return !out.empty();
}

}  // namespace

std::optional<nlohmann::json> lookupPath(
  const nlohmann::json& root, const std::string& dotted_path)
{
  if (!root.is_object()) {
    return std::nullopt;
  }
  // 扁平键优先：{"plan.send_to_control_ms": 1} 与嵌套两种写法都要认。
  {
    auto flat = root.find(dotted_path);
    if (flat != root.end()) {
      return flat.value();
    }
  }

  const nlohmann::json* node = &root;
  std::size_t start = 0;
  while (start <= dotted_path.size()) {
    const std::size_t dot = dotted_path.find('.', start);
    const std::string key = dotted_path.substr(
      start, dot == std::string::npos ? std::string::npos : dot - start);
    if (!node->is_object()) {
      return std::nullopt;
    }
    auto it = node->find(key);
    if (it == node->end()) {
      return std::nullopt;
    }
    if (dot == std::string::npos) {
      return it.value();
    }
    node = &it.value();
    start = dot + 1;
  }
  return std::nullopt;
}

bool WebCmd::poll()
{
  std::string raw;
  if (!readText(kPath, raw)) {
    return false;
  }
  if (raw == raw_signature_) {
    return true;
  }

  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(raw);
  } catch (...) {
    return false;
  }
  if (!parsed.is_object()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  command_ = std::move(parsed);
  raw_signature_ = std::move(raw);
  return true;
}

bool WebCmd::has(const std::string& dotted_path) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return lookupPath(command_, dotted_path).has_value();
}

nlohmann::json WebCmd::value(const std::string& dotted_path) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = lookupPath(command_, dotted_path);
  return found ? *found : nlohmann::json();
}

nlohmann::json WebCmd::snapshot() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return command_;
}

}  // namespace L6Telemetry
