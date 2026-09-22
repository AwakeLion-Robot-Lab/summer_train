#include "runtime/armor_detector_factory.hpp"

#include "l6_telemetry/logger.hpp"

#include <exception>
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
  L2Perception::InferenceModelConfig inference = config.inference;
  try {
    armor_backend->load(inference);
  } catch (const std::exception& error) {
    // GPU 插件或驱动（intel-opencl-icd）不在时 compile_model 会抛。退回 CPU 还
    // 能打，空检测器就只能看着，所以 OpenVINO 的非 CPU 设备失败时换 CPU 再试一次。
    const bool retry = config.inference_backend == L2Perception::InferenceBackendKind::OpenVino &&
                       !inference.device.starts_with("CPU");
    if (!retry) {
      throw;
    }
    L6Telemetry::logWarn(
      "armor model failed on", inference.device, error.what(), "; falling back to CPU");
    inference.device = "CPU";
    armor_backend->load(inference);
  }

  L2Perception::ArmorDecoderConfig decoder = config.decoder;
  const bool by_shape = guess_layout || config.auto_layout;
  if (by_shape) {
    decoder = L2Perception::decoderFor(L2Perception::probeOutputSpecs(*armor_backend));
    // 命令行临时换的模型未必配得上 YAML 的阈值，取预设的；YAML 自己写 auto 时
    // 阈值就是冲着这个模型写的，盖上去。
    if (!guess_layout) {
      applyThresholds(config.decoder_thresholds, decoder);
    }
  }
  const bool yolov8 =
    decoder.contract.tensor_layout == L2Perception::ArmorTensorLayout::FieldsByCandidates;

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
    "armor model loaded", backend_name, inference.model_path.string(), inference.device,
    "layout", yolov8 ? "yolov8_21" : "yolov5_22", by_shape ? "(by shape)" : "(yaml)",
    "output", decoder.contract.output_name, "conf", decoder.confidence_threshold,
    "number", config.number_classifier.enable
      ? config.number_classifier.model_path.string()
      : std::string{"off"});

  return detector;
}

}  // namespace runtime
