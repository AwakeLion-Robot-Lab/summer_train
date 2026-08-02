#include "l2_perception/inference/backends/tensorrt_backend.hpp"

#include <stdexcept>

namespace L2Perception
{

void TensorRtBackend::load(const InferenceModelConfig& config)
{
  (void)config;
  throw std::runtime_error(
    "TensorRT is not linked in this build. Keep the same IInferenceBackend contract and "
    "add the CUDA/TensorRT implementation when that SDK is installed.");
}

bool TensorRtBackend::ready() const noexcept
{
  return false;
}

const InferenceInputSpec& TensorRtBackend::inputSpec() const
{
  throw std::logic_error("TensorRtBackend is unavailable in this build");
}

InferenceResult TensorRtBackend::infer(const InferenceInput& input)
{
  (void)input;
  throw std::logic_error("TensorRtBackend is unavailable in this build");
}

}  // namespace L2Perception
