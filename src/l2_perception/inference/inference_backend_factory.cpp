#include "l2_perception/inference/inference_backend.hpp"

#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l2_perception/inference/backends/tensorrt_backend.hpp"

#include <cctype>
#include <stdexcept>
#include <string>

namespace L2Perception
{
namespace
{

[[nodiscard]] std::string normalizedBackendName(std::string_view name)
{
  std::string normalized;
  normalized.reserve(name.size());
  for (const unsigned char character : name) {
    if (std::isalnum(character) != 0) {
      normalized.push_back(static_cast<char>(std::tolower(character)));
    }
  }
  return normalized;
}

}  // namespace

std::string_view inferenceBackendName(InferenceBackendKind backend) noexcept
{
  switch (backend) {
    case InferenceBackendKind::OpenVino:
      return "openvino";
    case InferenceBackendKind::TensorRt:
      return "tensorrt";
  }
  return "unknown";
}

std::optional<InferenceBackendKind> inferenceBackendFromString(
  std::string_view name)
{
  const std::string normalized = normalizedBackendName(name);
  if (normalized == "openvino") {
    return InferenceBackendKind::OpenVino;
  }
  if (normalized == "tensorrt" || normalized == "trt") {
    return InferenceBackendKind::TensorRt;
  }
  return std::nullopt;
}

std::unique_ptr<IInferenceBackend> makeInferenceBackend(InferenceBackendKind backend)
{
  switch (backend) {
    case InferenceBackendKind::OpenVino:
      return std::make_unique<OpenVinoBackend>();
    case InferenceBackendKind::TensorRt:
      return std::make_unique<TensorRtBackend>();
  }
  throw std::invalid_argument("unknown inference backend");
}

}  // namespace L2Perception
