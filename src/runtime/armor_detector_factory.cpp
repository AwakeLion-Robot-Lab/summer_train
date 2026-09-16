#include "runtime/armor_detector_factory.hpp"

#include "l6_telemetry/logger.hpp"

#include <string>
#include <utility>

namespace runtime {

L2Perception::ArmorDetector makeDetector(const AutoAimConfig& config)
{
  const std::string backend_name{L2Perception::backendName(config.inference_backend)};

  // 模型路径、设备、颜色顺序、归一化和后端调度参数都在 inference 里，
  // loadConfig 已经把单列的 model_path/device 回填进去。宿主侧输入恒为
  // uint8 NHWC BGR，颜色顺序和归一化的转换在后端内部做。
  auto backend = L2Perception::makeBackend(config.inference_backend);
  backend->load(config.inference);

  L2Perception::NumberClassifier classifier;
  classifier.load(config.number_classifier);

  // 预处理用默认值：letterbox 左上贴齐、纯黑填充，与离线验证时一致。构造
  // ArmorDetector 时会核对灯条模型的输出形状。
  L2Perception::ArmorDetector detector(
    std::move(backend), std::move(classifier), config.light_decoder, config.light_matcher,
    config.light_finder);
  L6Telemetry::logInfo(
    "light model loaded", backend_name, config.inference.model_path.string(),
    config.inference.device, config.number_classifier.model_path.string());

  return detector;
}

}  // namespace runtime
