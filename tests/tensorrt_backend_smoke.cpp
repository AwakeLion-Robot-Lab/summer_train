#include "l2_perception/inference/backends/tensorrt_backend.hpp"
#include "runtime/auto_aim_config.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const auto runtime_config =
      runtime::loadAutoAimConfig("config/auto_aim.yaml");
    require(
      runtime_config.inference_backend ==
        L2Perception::InferenceBackendKind::TensorRt,
      "runtime config did not select TensorRT");
    require(
      runtime_config.model_path == "model/armor_model/0526.onnx" &&
        runtime_config.inference_device == "CUDA:0",
      "runtime TensorRT model/device config is wrong");
    require(
      std::abs(runtime_config.armor.small_width - 0.135) < 1e-12 &&
        std::abs(runtime_config.armor.big_width - 0.230) < 1e-12 &&
        std::abs(runtime_config.armor.height - 0.056) < 1e-12 &&
        std::abs(runtime_config.armor.corner_noise_px - 1.0) < 1e-12,
      "runtime L3 armor config is wrong");
    require(
      runtime_config.tracker.min_detect_count == 5 &&
        runtime_config.tracker.max_frame_interval.count() == 100 &&
        runtime_config.tracker.max_temp_lost_count == 15 &&
        runtime_config.tracker.outpost_max_temp_lost_count == 75,
      "runtime L3 tracker config is wrong");

    const auto backend_kind = L2Perception::inferenceBackendFromString("Tensor-RT");
    require(
      backend_kind == L2Perception::InferenceBackendKind::TensorRt,
      "TensorRT backend name was not parsed");
    require(
      !L2Perception::inferenceBackendFromString("unknown"),
      "an unknown inference backend name must be rejected");
    auto selected_backend = L2Perception::makeInferenceBackend(*backend_kind);
    require(
      dynamic_cast<L2Perception::TensorRtBackend*>(selected_backend.get()) != nullptr,
      "the inference backend factory selected the wrong implementation");

    L2Perception::TensorRtBackend backend;
    require(!backend.ready(), "a TensorRT backend must not be ready before load()");

    bool input_spec_threw = false;
    try {
      (void)backend.inputSpec();
    } catch (const std::logic_error&) {
      input_spec_threw = true;
    }
    require(input_spec_threw, "inputSpec() must reject use before load()");

#if defined(NEWVISION_HAS_TENSORRT)
    // Pass an engine/ONNX path as argv[1] to exercise real CUDA inference. Without a model,
    // keep this smoke test hardware-independent so it can still validate the public contract.
    if (argc < 2) {
      std::cout << "TensorRT backend interface smoke passed (model omitted)\n";
      return 0;
    }

    L2Perception::InferenceModelConfig config;
    config.model_path = argv[1];
    config.device = argc >= 3 ? argv[2] : "CUDA";
    backend.load(config);
    require(backend.ready(), "TensorRT backend did not become ready");

    const auto spec = backend.inputSpec();
    L2Perception::InferenceInput input;
    input.name = spec.name;
    input.shape = spec.shape;
    input.setOwnedData(std::vector<std::uint8_t>(input.elementCount(), 0));
    const auto result = backend.infer(input);
    require(!result.outputs.empty(), "TensorRT model produced no output tensors");
    for (const auto& output : result.outputs) {
      require(output.isConsistent(), "TensorRT output shape/data mismatch");
    }
    std::cout << "TensorRT backend smoke passed\n";
#else
    L2Perception::InferenceModelConfig config;
    config.model_path = "unused.onnx";
    bool load_threw = false;
    try {
      backend.load(config);
    } catch (const std::runtime_error& error) {
      load_threw = std::string{error.what()}.find("support is not enabled") != std::string::npos;
    }
    require(load_threw, "missing TensorRT support did not produce the configured error");
    std::cout << "TensorRT backend interface smoke passed (SDK omitted)\n";
#endif
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "TensorRT backend smoke failed: " << error.what() << '\n';
    return 1;
  }
}
