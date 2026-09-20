#include "runtime/armor_detector_factory.hpp"

#include "l6_telemetry/logger.hpp"

#include <string>
#include <utility>

namespace runtime {

L2Perception::ArmorDetector makeDetector(const AutoAimConfig& config, bool guess_layout)
{
  const std::string backend_name{L2Perception::backendName(config.inference_backend)};

  // 模型路径、设备、颜色顺序、归一化和后端调度参数都在 inference 里，
  // loadConfig 已经把单列的 model_path/device 回填进去。宿主侧输入恒为
  // uint8 NHWC BGR，颜色顺序和归一化的转换在后端内部做。
  auto armor_backend = L2Perception::makeBackend(config.inference_backend);
  armor_backend->load(config.inference);

  L2Perception::ArmorDecoderConfig decoder = config.decoder;
  if (guess_layout) {
    decoder = L2Perception::decoderFor(L2Perception::probeOutputSpecs(*armor_backend));
  }

  // 预处理用默认值：letterbox 左上贴齐、纯黑填充，与离线验证时一致。构造
  // ArmorDetector 时会核对整板模型的输出形状。
  L2Perception::ArmorDetector detector(
    std::move(armor_backend),
    L2Perception::ArmorDetectorConfig{
      .decoder = decoder,
      .refiner = config.refiner,
      .finder = config.light_finder,
      .number = config.number_classifier});
  L6Telemetry::logInfo(
    "armor model loaded", backend_name, config.inference.model_path.string(),
    config.inference.device, "output", decoder.contract.output_name,
    "number", config.number_classifier.enable
      ? config.number_classifier.model_path.string()
      : std::string{"off"});

  return detector;
}

}  // namespace runtime
