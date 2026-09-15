#include "runtime/armor_detector_factory.hpp"

#include "l6_telemetry/logger.hpp"

#include <string>
#include <utility>

namespace runtime {

L2Perception::ArmorDetector makeDetector(const AutoAimConfig& config)
{
  const std::string backend_name{L2Perception::backendName(config.inference_backend)};

  // 模型路径、设备、颜色顺序、归一化以及后端调度参数全部来自 inference 节点，
  // loadConfig 已经把 model_path/device 回填进去。宿主输入恒为 uint8
  // NHWC BGR；颜色和归一化转换由具体后端完成。
  auto backend = L2Perception::makeBackend(config.inference_backend);
  backend->load(config.inference);

  L2Perception::NumberClassifier classifier;
  classifier.load(config.number_classifier);

  // 预处理保持默认（letterbox 左上贴齐、纯黑填充，与离线验证时一致）。构造时
  // 会核对灯条模型的输出形状。
  L2Perception::ArmorDetector detector(
    std::move(backend), std::move(classifier), config.light_decoder, config.light_matcher);
  L6Telemetry::logInfo(
    "light model loaded", backend_name, config.inference.model_path.string(),
    config.inference.device, config.number_classifier.model_path.string());

  return detector;
}

}  // namespace runtime
