#include "l2_perception/inference/backends/tensorrt_backend.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(NEWVISION_HAS_TENSORRT)
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#if NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
#error "TensorRtBackend requires TensorRT 8.5 or newer"
#endif
#endif

namespace L2Perception
{

#if defined(NEWVISION_HAS_TENSORRT)
namespace
{

class TensorRtLogger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char* message) noexcept override
  {
    // TensorRT 的 INFO/VERBOSE 日志非常密集；L2 启动阶段只保留 warning 以上，
    // 解析 ONNX 失败时仍能从异常和这里的诊断同时定位问题。
    if (severity <= Severity::kWARNING) {
      try {
        std::lock_guard lock(mutex_);
        std::cerr << "[TensorRT] " << (message == nullptr ? "" : message) << '\n';
      } catch (...) {
        // ILogger::log() 是 noexcept；诊断输出失败不能终止推理进程。
      }
    }
  }

private:
  std::mutex mutex_;
};

TensorRtLogger g_tensor_rt_logger;

template<typename Type>
struct TensorRtDeleter
{
  void operator()(Type* object) const noexcept
  {
    delete object;
  }
};

template<typename Type>
using TensorRtPtr = std::unique_ptr<Type, TensorRtDeleter<Type>>;

std::string cudaErrorText(cudaError_t error)
{
  const char* message = cudaGetErrorString(error);
  return message == nullptr ? "unknown CUDA error" : std::string{message};
}

void checkCuda(cudaError_t error, std::string_view operation)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(
      "TensorRtBackend " + std::string{operation} + " failed: " + cudaErrorText(error));
  }
}

class CudaDeviceGuard
{
public:
  explicit CudaDeviceGuard(int requested_device)
  {
    checkCuda(cudaGetDevice(&previous_device_), "cudaGetDevice");
    if (previous_device_ != requested_device) {
      checkCuda(cudaSetDevice(requested_device), "cudaSetDevice");
      restore_previous_device_ = true;
    }
  }

  ~CudaDeviceGuard() noexcept
  {
    if (restore_previous_device_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

private:
  int previous_device_{0};
  bool restore_previous_device_{false};
};

std::string lowerExtension(const std::filesystem::path& path)
{
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return extension;
}

int parseCudaDevice(std::string_view configured_device)
{
  const std::string device{configured_device};
  std::string upper = device;
  std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });

  std::string index_text;
  if (upper == "CUDA" || upper == "GPU") {
    return 0;
  }
  if (upper.starts_with("CUDA:") || upper.starts_with("GPU:")) {
    index_text = upper.substr(upper.find(':') + 1);
  } else if (!upper.empty() && std::all_of(upper.begin(), upper.end(), [](unsigned char character) {
               return std::isdigit(character) != 0;
             })) {
    index_text = upper;
  } else {
    throw std::invalid_argument(
      "TensorRtBackend device must be CUDA, CUDA:<index>, GPU, or GPU:<index>; got " + device);
  }

  if (index_text.empty()) {
    throw std::invalid_argument("TensorRtBackend received an empty CUDA device index");
  }

  try {
    std::size_t parsed_length = 0;
    const long long index = std::stoll(index_text, &parsed_length);
    if (parsed_length != index_text.size() || index < 0
        || index > static_cast<long long>(std::numeric_limits<int>::max())) {
      throw std::out_of_range("CUDA device index out of range");
    }
    return static_cast<int>(index);
  } catch (const std::exception&) {
    throw std::invalid_argument("TensorRtBackend received an invalid CUDA device: " + device);
  }
}

std::size_t dataTypeSize(nvinfer1::DataType data_type)
{
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(std::uint16_t);
    default:
      throw std::runtime_error("TensorRtBackend only supports FP32 and FP16 tensors");
  }
}

std::vector<std::size_t> staticShape(
  const nvinfer1::Dims& dims, std::string_view tensor_name)
{
  if (dims.nbDims <= 0) {
    throw std::runtime_error(
      "TensorRtBackend tensor has no dimensions: " + std::string{tensor_name});
  }

  std::vector<std::size_t> shape;
  shape.reserve(static_cast<std::size_t>(dims.nbDims));
  for (int index = 0; index < dims.nbDims; ++index) {
    if (dims.d[index] <= 0) {
      throw std::runtime_error(
        "TensorRtBackend requires static tensor shapes; dynamic dimension found in "
        + std::string{tensor_name});
    }
    const auto dimension = static_cast<std::uint64_t>(dims.d[index]);
    if (dimension > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("TensorRtBackend tensor dimension exceeds host size_t");
    }
    shape.push_back(static_cast<std::size_t>(dimension));
  }
  return shape;
}

std::size_t shapeElementCount(
  const std::vector<std::size_t>& shape, std::string_view tensor_name)
{
  std::size_t count = 1;
  for (const std::size_t dimension : shape) {
    if (dimension == 0 || count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error(
        "TensorRtBackend tensor element count overflows size_t: " + std::string{tensor_name});
    }
    count *= dimension;
  }
  return count;
}

std::size_t byteCount(
  std::size_t element_count, nvinfer1::DataType data_type, std::string_view tensor_name)
{
  const std::size_t element_size = dataTypeSize(data_type);
  if (element_count > std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::overflow_error(
      "TensorRtBackend tensor byte count overflows size_t: " + std::string{tensor_name});
  }
  return element_count * element_size;
}

void requireLinearIoFormat(const nvinfer1::ICudaEngine& engine, const char* tensor_name)
{
  if (engine.getTensorFormat(tensor_name) == nvinfer1::TensorFormat::kLINEAR) {
    return;
  }

  const char* description = engine.getTensorFormatDesc(tensor_name);
  throw std::runtime_error(
    "TensorRtBackend requires linear I/O tensors; " + std::string{tensor_name}
    + " uses " + (description == nullptr ? "an unknown TensorRT format" : description));
}

TensorRtPtr<nvinfer1::ICudaEngine> deserializeEngine(
  nvinfer1::IRuntime& runtime, const std::filesystem::path& engine_path)
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error(
      "TensorRtBackend could not open engine file: " + engine_path.string());
  }

  const std::streamsize file_size = file.tellg();
  if (file_size <= 0) {
    throw std::runtime_error("TensorRtBackend received an empty engine file: " + engine_path.string());
  }
  file.seekg(0, std::ios::beg);

  std::vector<char> serialized(static_cast<std::size_t>(file_size));
  if (!file.read(serialized.data(), file_size)) {
    throw std::runtime_error(
      "TensorRtBackend could not read engine file: " + engine_path.string());
  }

  auto* engine = runtime.deserializeCudaEngine(serialized.data(), serialized.size());
  if (engine == nullptr) {
    throw std::runtime_error(
      "TensorRtBackend failed to deserialize TensorRT engine: " + engine_path.string());
  }
  return TensorRtPtr<nvinfer1::ICudaEngine>{engine};
}

TensorRtPtr<nvinfer1::ICudaEngine> buildEngineFromOnnx(
  nvinfer1::IRuntime& runtime, const std::filesystem::path& onnx_path)
{
  auto builder = TensorRtPtr<nvinfer1::IBuilder>{
    nvinfer1::createInferBuilder(g_tensor_rt_logger)};
  if (!builder) {
    throw std::runtime_error("TensorRtBackend failed to create TensorRT builder");
  }

  // TensorRT 10 removed kEXPLICIT_BATCH because explicit batch is now the only mode. TensorRT
  // 10 uses kSTRONGLY_TYPED for ONNX networks, while TensorRT 11 is strongly typed by default.
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 11
  const auto network_flags = 0U;
#elif defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
  const auto network_flags =
    1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#else
  const auto network_flags =
    1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
  auto network = TensorRtPtr<nvinfer1::INetworkDefinition>{
    builder->createNetworkV2(network_flags)};
  if (!network) {
    throw std::runtime_error("TensorRtBackend failed to create TensorRT network");
  }

  auto parser = TensorRtPtr<nvonnxparser::IParser>{
    nvonnxparser::createParser(*network, g_tensor_rt_logger)};
  if (!parser) {
    throw std::runtime_error("TensorRtBackend failed to create ONNX parser");
  }
  if (!parser->parseFromFile(
        onnx_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::string details = "TensorRtBackend failed to parse ONNX model: " + onnx_path.string();
    for (int index = 0; index < parser->getNbErrors(); ++index) {
      const auto* error = parser->getError(index);
      if (error != nullptr && error->desc() != nullptr) {
        details += "\n  " + std::string{error->desc()};
      }
    }
    throw std::runtime_error(details);
  }

  if (network->getNbInputs() != 1) {
    throw std::runtime_error("TensorRtBackend currently supports exactly one ONNX input");
  }

  // L2 allocates tightly packed logical tensors and preprocesses the input as contiguous NCHW.
  // Restricting ONNX-built engines to kLINEAR prevents TensorRT from selecting vectorized formats
  // such as CHW2/HWC8, whose padding and component layout require a different buffer calculation.
  const auto linear_format =
    1U << static_cast<std::uint32_t>(nvinfer1::TensorFormat::kLINEAR);
  for (int index = 0; index < network->getNbInputs(); ++index) {
    network->getInput(index)->setAllowedFormats(linear_format);
  }
  for (int index = 0; index < network->getNbOutputs(); ++index) {
    network->getOutput(index)->setAllowedFormats(linear_format);
  }

  auto builder_config = TensorRtPtr<nvinfer1::IBuilderConfig>{builder->createBuilderConfig()};
  if (!builder_config) {
    throw std::runtime_error("TensorRtBackend failed to create TensorRT builder config");
  }

  // The engine is built once during startup. One GiB is deliberately a limit rather than a
  // request to reserve the whole amount; TensorRT only uses the workspace it actually needs.
  builder_config->setMemoryPoolLimit(
    nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<std::size_t>(1ULL << 30));
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
  if (builder->platformHasFastFp16()) {
    // This allows TensorRT to select FP16 kernels while preserving the model's I/O types.
    // The inference path below still converts FP16 I/O to the L2 float32 contract.
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
#endif

  auto serialized = TensorRtPtr<nvinfer1::IHostMemory>{
    builder->buildSerializedNetwork(*network, *builder_config)};
  if (!serialized) {
    throw std::runtime_error(
      "TensorRtBackend failed to build a serialized engine from ONNX: " + onnx_path.string());
  }

  auto* engine = runtime.deserializeCudaEngine(serialized->data(), serialized->size());
  if (engine == nullptr) {
    throw std::runtime_error("TensorRtBackend failed to deserialize the ONNX-built engine");
  }
  return TensorRtPtr<nvinfer1::ICudaEngine>{engine};
}

std::uint16_t floatToHalf(float value) noexcept
{
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t exponent = (bits >> 23U) & 0xffU;
  std::uint32_t mantissa = bits & 0x7fffffU;

  if (exponent == 0xffU) {
    return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00U : 0x7e00U));
  }

  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }

    mantissa |= 0x800000U;
    const int shift = 14 - half_exponent;
    std::uint32_t half_mantissa = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0U)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }

  std::uint32_t half_mantissa = mantissa >> 13U;
  const std::uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0U)) {
    ++half_mantissa;
  }
  if (half_mantissa == 0x400U) {
    half_mantissa = 0;
    ++half_exponent;
    if (half_exponent >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10U)
                                    | half_mantissa);
}

float halfToFloat(std::uint16_t value) noexcept
{
  const std::uint32_t sign = (value >> 15U) & 1U;
  const std::uint32_t exponent = (value >> 10U) & 0x1fU;
  const std::uint32_t mantissa = value & 0x3ffU;

  float result = 0.0F;
  if (exponent == 0) {
    if (mantissa != 0) {
      result = std::ldexp(static_cast<float>(mantissa), -24);
    }
  } else if (exponent == 0x1fU) {
    result = mantissa == 0 ? std::numeric_limits<float>::infinity()
                           : std::numeric_limits<float>::quiet_NaN();
  } else {
    result = std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F,
                        static_cast<int>(exponent) - 15);
  }

  return sign == 0 ? result : -result;
}

}  // namespace

struct TensorRtBackend::Impl
{
  struct OutputBinding
  {
    std::string name;
    std::vector<std::size_t> shape;
    std::size_t element_count{0};
    std::size_t byte_count{0};
    nvinfer1::DataType data_type{nvinfer1::DataType::kFLOAT};
    void* device_buffer{nullptr};
  };

  TensorRtPtr<nvinfer1::IRuntime> runtime;
  TensorRtPtr<nvinfer1::ICudaEngine> engine;
  TensorRtPtr<nvinfer1::IExecutionContext> context;

  std::mutex inference_mutex;
  cudaStream_t stream{nullptr};
  void* device_input{nullptr};
  std::vector<OutputBinding> outputs;

  int device_index{0};
  std::string input_name;
  nvinfer1::DataType input_data_type{nvinfer1::DataType::kFLOAT};
  std::size_t input_height{0};
  std::size_t input_width{0};
  std::size_t input_byte_count{0};
  float normalization_divisor{255.0F};
  ModelColorOrder model_color_order{ModelColorOrder::Rgb};
  std::vector<float> host_input_float;
  std::vector<std::uint16_t> host_input_half;

  ~Impl() noexcept { release(); }

  void release() noexcept
  {
    // TensorRT/CUDA objects are tied to the CUDA device on which they were created. Preserve the
    // caller's current device while releasing them; this matters when a backend is reloaded from
    // CUDA:0 to CUDA:1 or when another subsystem owns the CUDA device selection.
    int previous_device = -1;
    const bool had_current_device = cudaGetDevice(&previous_device) == cudaSuccess;
    (void)cudaSetDevice(device_index);
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }

    if (device_input != nullptr) {
      (void)cudaFree(device_input);
      device_input = nullptr;
    }
    for (auto& output : outputs) {
      if (output.device_buffer != nullptr) {
        (void)cudaFree(output.device_buffer);
        output.device_buffer = nullptr;
      }
    }

    context.reset();
    engine.reset();
    runtime.reset();

    if (had_current_device && previous_device != device_index) {
      (void)cudaSetDevice(previous_device);
    }
  }
};

#else

// Keep the public class linkable on machines without CUDA/TensorRT. This is important for the
// normal CPU/OpenVINO build: merely including the backend header must not make TensorRT a hard
// dependency.
struct TensorRtBackend::Impl
{
};

#endif

TensorRtBackend::TensorRtBackend()
  : impl_(std::make_unique<Impl>())
{
}

TensorRtBackend::~TensorRtBackend() = default;
TensorRtBackend::TensorRtBackend(TensorRtBackend&&) noexcept = default;
TensorRtBackend& TensorRtBackend::operator=(TensorRtBackend&&) noexcept = default;

void TensorRtBackend::load(const InferenceModelConfig& config)
{
#if defined(NEWVISION_HAS_TENSORRT)
  if (config.model_path.empty()) {
    throw std::invalid_argument("TensorRtBackend requires a model_path");
  }
  if (!std::filesystem::is_regular_file(config.model_path)) {
    throw std::runtime_error(
      "TensorRtBackend model file does not exist: " + config.model_path.string());
  }
  if (!std::isfinite(config.normalization_divisor) || config.normalization_divisor <= 0.0F) {
    throw std::invalid_argument("TensorRtBackend requires a positive normalization divisor");
  }

  auto next = std::make_unique<Impl>();
  next->device_index = parseCudaDevice(config.device);
  const CudaDeviceGuard device_guard(next->device_index);

  next->runtime.reset(nvinfer1::createInferRuntime(g_tensor_rt_logger));
  if (!next->runtime) {
    throw std::runtime_error("TensorRtBackend failed to create TensorRT runtime");
  }

  const std::string extension = lowerExtension(config.model_path);
  TensorRtPtr<nvinfer1::ICudaEngine> engine;
  if (extension == ".onnx") {
    engine = buildEngineFromOnnx(*next->runtime, config.model_path);
  } else {
    // Serialized TensorRT plans are commonly named .engine or .plan, but accepting any
    // non-ONNX suffix also supports deployment systems that use a versioned filename.
    engine = deserializeEngine(*next->runtime, config.model_path);
  }
  if (!engine) {
    throw std::runtime_error("TensorRtBackend did not obtain a TensorRT engine");
  }
  next->engine = std::move(engine);

  if (next->engine->getNbIOTensors() < 2) {
    throw std::runtime_error("TensorRtBackend requires one input and at least one output tensor");
  }

  for (int index = 0; index < next->engine->getNbIOTensors(); ++index) {
    const char* tensor_name = next->engine->getIOTensorName(index);
    if (tensor_name == nullptr || *tensor_name == '\0') {
      throw std::runtime_error("TensorRtBackend encountered an unnamed I/O tensor");
    }
    const std::string name{tensor_name};
    requireLinearIoFormat(*next->engine, tensor_name);
    if (next->engine->getTensorLocation(tensor_name) != nvinfer1::TensorLocation::kDEVICE) {
      throw std::runtime_error(
        "TensorRtBackend only supports device I/O tensors: " + name);
    }

    const auto io_mode = next->engine->getTensorIOMode(tensor_name);
    if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
      if (!next->input_name.empty()) {
        throw std::runtime_error("TensorRtBackend currently supports exactly one model input");
      }
      const auto shape = staticShape(next->engine->getTensorShape(tensor_name), name);
      if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] == 0
          || shape[3] == 0) {
        throw std::runtime_error(
          "TensorRtBackend requires model input shape [1, 3, H, W]: " + name);
      }

      const auto data_type = next->engine->getTensorDataType(tensor_name);
      if (data_type != nvinfer1::DataType::kFLOAT && data_type != nvinfer1::DataType::kHALF) {
        throw std::runtime_error("TensorRtBackend model input must be FP32 or FP16: " + name);
      }
      next->input_name = name;
      next->input_data_type = data_type;
      next->input_height = shape[2];
      next->input_width = shape[3];
      next->input_byte_count = byteCount(shapeElementCount(shape, name), data_type, name);
    } else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
      OutputBinding output;
      output.name = name;
      output.shape = staticShape(next->engine->getTensorShape(tensor_name), name);
      output.element_count = shapeElementCount(output.shape, name);
      output.data_type = next->engine->getTensorDataType(tensor_name);
      if (output.data_type != nvinfer1::DataType::kFLOAT
          && output.data_type != nvinfer1::DataType::kHALF) {
        throw std::runtime_error("TensorRtBackend model output must be FP32 or FP16: " + name);
      }
      output.byte_count = byteCount(output.element_count, output.data_type, name);
      next->outputs.push_back(std::move(output));
    }
  }

  if (next->input_name.empty() || next->outputs.empty()) {
    throw std::runtime_error("TensorRtBackend engine has an invalid input/output tensor set");
  }

  next->context.reset(next->engine->createExecutionContext());
  if (!next->context) {
    throw std::runtime_error("TensorRtBackend failed to create an execution context");
  }
  checkCuda(cudaStreamCreate(&next->stream), "cudaStreamCreate");
  checkCuda(cudaMalloc(&next->device_input, next->input_byte_count), "cudaMalloc(input)");
  for (auto& output : next->outputs) {
    checkCuda(cudaMalloc(&output.device_buffer, output.byte_count), "cudaMalloc(output)");
  }

  next->normalization_divisor = config.normalization_divisor;
  next->model_color_order = config.model_color_order;
  if (next->input_data_type == nvinfer1::DataType::kFLOAT) {
    next->host_input_float.resize(next->input_byte_count / sizeof(float));
  } else {
    next->host_input_half.resize(next->input_byte_count / sizeof(std::uint16_t));
  }

  InferenceInputSpec next_input_spec;
  next_input_spec.name = next->input_name;
  next_input_spec.shape = {1, next->input_height, next->input_width, 3};

  // Publish only after every TensorRT/CUDA resource has been created successfully. A failed
  // reload therefore keeps the previous model usable instead of exposing partial state.
  impl_ = std::move(next);
  input_spec_ = std::move(next_input_spec);
  ready_ = true;
#else
  (void)config;
  throw std::runtime_error(
    "TensorRT support is not enabled; configure with xmake f --use_tensorrt=y "
    "and provide CUDA/TensorRT SDK paths");
#endif
}

bool TensorRtBackend::ready() const noexcept
{
#if defined(NEWVISION_HAS_TENSORRT)
  return ready_ && impl_ != nullptr && impl_->engine != nullptr && impl_->context != nullptr;
#else
  return false;
#endif
}

const InferenceInputSpec& TensorRtBackend::inputSpec() const
{
  if (!ready()) {
    throw std::logic_error("TensorRtBackend inputSpec() called before load()");
  }
  return input_spec_;
}

InferenceResult TensorRtBackend::infer(const InferenceInput& input)
{
  if (!ready()) {
    throw std::logic_error("TensorRtBackend infer() called before load()");
  }
  if (input.shape != input_spec_.shape || !input.isConsistent()) {
    throw std::invalid_argument("TensorRtBackend received an input tensor with an invalid shape");
  }

#if defined(NEWVISION_HAS_TENSORRT)
  std::unique_lock lock(impl_->inference_mutex);
  const CudaDeviceGuard device_guard(impl_->device_index);

  const std::span<const std::uint8_t> input_values = input.values();
  const std::size_t plane_size = impl_->input_height * impl_->input_width;
  const bool swap_red_blue = impl_->model_color_order == ModelColorOrder::Rgb;
  const auto sourceValue = [&](std::size_t row, std::size_t column, std::size_t channel) {
    const std::size_t source_channel = swap_red_blue ? 2U - channel : channel;
    return static_cast<float>(input_values[(row * impl_->input_width + column) * 3U
                                            + source_channel])
           / impl_->normalization_divisor;
  };

  if (impl_->input_data_type == nvinfer1::DataType::kFLOAT) {
    for (std::size_t row = 0; row < impl_->input_height; ++row) {
      for (std::size_t column = 0; column < impl_->input_width; ++column) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
          impl_->host_input_float[channel * plane_size + row * impl_->input_width + column] =
            sourceValue(row, column, channel);
        }
      }
    }
    checkCuda(cudaMemcpyAsync(
                impl_->device_input, impl_->host_input_float.data(), impl_->input_byte_count,
                cudaMemcpyHostToDevice, impl_->stream),
              "cudaMemcpyAsync(input)");
  } else {
    for (std::size_t row = 0; row < impl_->input_height; ++row) {
      for (std::size_t column = 0; column < impl_->input_width; ++column) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
          impl_->host_input_half[channel * plane_size + row * impl_->input_width + column] =
            floatToHalf(sourceValue(row, column, channel));
        }
      }
    }
    checkCuda(cudaMemcpyAsync(
                impl_->device_input, impl_->host_input_half.data(), impl_->input_byte_count,
                cudaMemcpyHostToDevice, impl_->stream),
              "cudaMemcpyAsync(input)");
  }

  if (!impl_->context->setTensorAddress(impl_->input_name.c_str(), impl_->device_input)) {
    throw std::runtime_error("TensorRtBackend failed to bind input tensor: " + impl_->input_name);
  }
  for (const auto& output : impl_->outputs) {
    if (!impl_->context->setTensorAddress(output.name.c_str(), output.device_buffer)) {
      throw std::runtime_error("TensorRtBackend failed to bind output tensor: " + output.name);
    }
  }
  if (!impl_->context->enqueueV3(impl_->stream)) {
    throw std::runtime_error("TensorRtBackend inference failed at enqueueV3");
  }

  struct HostOutput
  {
    std::vector<float> float_values;
    std::vector<std::uint16_t> half_values;
  };
  std::vector<HostOutput> host_outputs(impl_->outputs.size());
  for (std::size_t index = 0; index < impl_->outputs.size(); ++index) {
    const auto& output = impl_->outputs[index];
    if (output.data_type == nvinfer1::DataType::kFLOAT) {
      host_outputs[index].float_values.resize(output.element_count);
      checkCuda(cudaMemcpyAsync(
                  host_outputs[index].float_values.data(), output.device_buffer,
                  output.byte_count, cudaMemcpyDeviceToHost, impl_->stream),
                "cudaMemcpyAsync(output)");
    } else {
      host_outputs[index].half_values.resize(output.element_count);
      checkCuda(cudaMemcpyAsync(
                  host_outputs[index].half_values.data(), output.device_buffer,
                  output.byte_count, cudaMemcpyDeviceToHost, impl_->stream),
                "cudaMemcpyAsync(output)");
    }
  }
  checkCuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");

  InferenceResult result;
  result.outputs.reserve(impl_->outputs.size());
  for (std::size_t index = 0; index < impl_->outputs.size(); ++index) {
    const auto& output = impl_->outputs[index];
    InferenceTensor tensor;
    tensor.name = output.name;
    tensor.shape = output.shape;
    if (output.data_type == nvinfer1::DataType::kFLOAT) {
      tensor.setOwnedData(std::move(host_outputs[index].float_values));
    } else {
      std::vector<float> values;
      values.resize(output.element_count);
      for (std::size_t element = 0; element < output.element_count; ++element) {
        values[element] = halfToFloat(host_outputs[index].half_values[element]);
      }
      tensor.setOwnedData(std::move(values));
    }
    result.outputs.push_back(std::move(tensor));
  }
  return result;
#else
  (void)input;
  throw std::runtime_error("TensorRT support is not enabled");
#endif
}

}  // namespace L2Perception
