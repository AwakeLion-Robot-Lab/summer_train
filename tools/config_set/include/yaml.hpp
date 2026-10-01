#pragma once

#include "l6_telemetry/logger.hpp"

#include <string>
#include <yaml-cpp/yaml.h>

namespace tools {

inline YAML::Node load(const std::string& path)
{
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::BadFile& e) {
    LOG_ERROR("[YAML] Failed to load file:", e.what());
    std::exit(1);
  } catch (const YAML::ParserException& e) {
    LOG_ERROR("[YAML] Parser error:", e.what());
    std::exit(1);
  }
}

template <typename T>
inline T read(const YAML::Node& yaml, const std::string& key)
{
  if (yaml[key]) {
    return yaml[key].as<T>();
  }
  LOG_ERROR("[YAML]", key, "not found!");
  std::exit(1);
}

}  // namespace tools
