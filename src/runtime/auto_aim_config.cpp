#include "runtime/auto_aim_config.hpp"

#include "l6_telemetry/logger.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace runtime {

AutoAimConfig loadAutoAimConfig(const std::string& path)
{
  AutoAimConfig config;
  if (!std::filesystem::exists(path)) {
    L6Telemetry::logWarn("auto-aim config not found, using defaults", path);
    return config;
  }

  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node inference = root["inference"];
  if (!inference) {
    L6Telemetry::logWarn("auto-aim inference config missing, using defaults", path);
    return config;
  }

  if (inference["backend"]) {
    const std::string name = inference["backend"].as<std::string>();
    const auto backend = L2Perception::inferenceBackendFromString(name);
    if (!backend) {
      throw std::runtime_error(
        "inference.backend must be 'openvino' or 'tensorrt'; got " + name);
    }
    config.inference_backend = *backend;
  }
  if (inference["model_path"]) {
    config.model_path = inference["model_path"].as<std::string>();
  }
  if (inference["device"]) {
    config.inference_device = inference["device"].as<std::string>();
  }

  L6Telemetry::logInfo(
    "inference config loaded", path,
    std::string{L2Perception::inferenceBackendName(config.inference_backend)},
    config.model_path.string(), config.inference_device);
  return config;
}

}  // namespace runtime
