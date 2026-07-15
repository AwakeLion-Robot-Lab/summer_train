#pragma once

#include "l2_perception/inference/inference_backend.hpp"

#include <memory>

namespace L2Perception
{

// OpenVINO 接收通用 U8 NHWC BGR letterbox，在预处理图中完成颜色、归一化、
// NHWC->NCHW 以及到原始模型 FP16/FP32 输入类型的转换。
class OpenVinoBackend final : public IInferenceBackend
{
public:
  OpenVinoBackend();
  ~OpenVinoBackend() override;

  OpenVinoBackend(OpenVinoBackend&&) noexcept;
  OpenVinoBackend& operator=(OpenVinoBackend&&) noexcept;
  OpenVinoBackend(const OpenVinoBackend&) = delete;
  OpenVinoBackend& operator=(const OpenVinoBackend&) = delete;

  void load(const InferenceModelConfig& config) override;
  [[nodiscard]] bool ready() const noexcept override;
  [[nodiscard]] const InferenceInputSpec& inputSpec() const override;
  [[nodiscard]] InferenceResult infer(const InferenceInput& input) override;

private:
  struct Impl;
  // Impl 内的共享请求池可被结果租约保活，Backend 先析构时输出缓冲区仍然有效。
  std::shared_ptr<Impl> impl_;
  InferenceInputSpec input_spec_;
  bool ready_{false};
};

}  // namespace L2Perception
