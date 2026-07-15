#pragma once

#include "l2_perception/inference/inference_backend.hpp"

namespace L2Perception
{

// TensorRT 的公开边界与 OpenVINO 完全一致。当前工程未链接 CUDA/TensorRT，
// 因而此类会在 load() 时给出明确错误；以后只替换 .cpp，不影响 L2 调用方。
class TensorRtBackend final : public IInferenceBackend
{
public:
  void load(const InferenceModelConfig& config) override;
  [[nodiscard]] bool ready() const noexcept override;
  [[nodiscard]] const InferenceInputSpec& inputSpec() const override;
  [[nodiscard]] InferenceResult infer(const InferenceInput& input) override;

private:
  InferenceInputSpec input_spec_;
};

}  // namespace L2Perception
