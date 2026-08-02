#include "l2_perception/inference/backends/openvino_backend.hpp"

#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(NEWVISION_HAS_OPENVINO)
#include <openvino/openvino.hpp>
#endif

namespace L2Perception
{

struct OpenVinoBackend::Impl
{
#if defined(NEWVISION_HAS_OPENVINO)
  // SDK 类型被放在 PIMPL 内，公开头文件无需包含 OpenVINO，减少依赖传播。
  ov::Core core;
  struct RequestPool
  {
    explicit RequestPool(ov::CompiledModel compiled_model)
      : compiled_model(std::move(compiled_model))
    {
      // 正常同步链路只需要一个请求；提前创建可避免第一帧临时分配。
      idle_requests.reserve(max_cached_requests);
      idle_requests.push_back(this->compiled_model.create_infer_request());
    }

    ov::InferRequest acquire()
    {
      {
        std::lock_guard lock(mutex);
        if (!idle_requests.empty()) {
          //像栈一样
          ov::InferRequest request = std::move(idle_requests.back());
          idle_requests.pop_back();
          return request;
        }
      }

      throw std::runtime_error(
        "OpenVINO infer request is busy; queued inference is disabled");
    }

    void release(ov::InferRequest&& request) noexcept
    {
      try {
        std::lock_guard lock(mutex);
        if (idle_requests.size() < max_cached_requests) {
          idle_requests.push_back(std::move(request));
        }
      } catch (...) {
        // 释放结果不能抛异常；缓存失败时直接让 request 正常析构即可。
      }
    }

    static constexpr std::size_t max_cached_requests = 1;
    ov::CompiledModel compiled_model;
    std::mutex mutex;
    std::vector<ov::InferRequest> idle_requests;
  };

  std::shared_ptr<RequestPool> request_pool;
  std::vector<std::string> output_names;

  // 每个 ResultLease 独占一个 InferRequest。只要结果仍被 Decoder 持有，该请求及其输出
  // 缓冲区就不会被下一帧复用；下一帧使用另一个请求，因此也不会发生跨帧自锁。
  struct ResultLease
  {
    explicit ResultLease(std::shared_ptr<RequestPool> pool)
      : pool(std::move(pool))
      , infer_request(this->pool->acquire())
    {
    }

    ~ResultLease()
    {
      // 先释放额外的 Tensor 句柄，再把拥有同一输出缓冲区的请求交还池中。
      output_tensors.clear();
      pool->release(std::move(infer_request));
    }

    std::shared_ptr<RequestPool> pool;
    ov::InferRequest infer_request;
    std::vector<ov::Tensor> output_tensors;
  };
#endif
};

OpenVinoBackend::OpenVinoBackend()
  : impl_(std::make_shared<Impl>())
{
}

OpenVinoBackend::~OpenVinoBackend() = default;
OpenVinoBackend::OpenVinoBackend(OpenVinoBackend&&) noexcept = default;
OpenVinoBackend& OpenVinoBackend::operator=(OpenVinoBackend&&) noexcept = default;

void OpenVinoBackend::load(const InferenceModelConfig& config)
{
#if defined(NEWVISION_HAS_OPENVINO)
  if (!impl_) {
    // 支持对 move 后的有效对象重新 load()；ready() 会把无 Impl 的对象视为未就绪。
    impl_ = std::make_shared<Impl>();
  }
  ready_ = false;

  if (config.model_path.empty()) {
    throw std::invalid_argument("OpenVinoBackend requires a model_path");
  }
  if (!std::filesystem::exists(config.model_path)) {
    throw std::runtime_error("OpenVINO model file does not exist: " + config.model_path.string());
  }

  // 读入 ONNX 或 OpenVINO IR（XML + 同名 BIN），此时尚未针对设备编译。
  auto model = impl_->core.read_model(config.model_path.string());
  if (model->inputs().size() != 1) {
    throw std::runtime_error("OpenVinoBackend currently supports exactly one model input");
  }
  if (model->input().get_partial_shape().is_dynamic()) {
    throw std::runtime_error(
      "OpenVinoBackend requires a static input model; export a static model before loading it");
  }
  if (!std::isfinite(config.normalization_divisor) || config.normalization_divisor <= 0.0F) {
    throw std::invalid_argument("OpenVinoBackend requires a positive normalization divisor");
  }

  // 先按原始模型契约取 NCHW 尺寸和数据类型。armor.xml 是 FP16，buff.xml 可以是 FP32；
  // 宿主侧不必跟随它们，后续 PrePostProcessor 会统一暴露 U8 NHWC 输入。
  const auto original_input = model->input();
  const ov::element::Type original_input_type = original_input.get_element_type();
  if (original_input_type != ov::element::f16 && original_input_type != ov::element::f32) {
    throw std::runtime_error("OpenVinoBackend only supports FP16 or FP32 image models");
  }

  const ov::Shape original_input_shape = original_input.get_shape();
  if (original_input_shape.size() != 4 || original_input_shape[0] != 1
      || original_input_shape[1] != 3 || original_input_shape[2] == 0
      || original_input_shape[3] == 0) {
    throw std::runtime_error(
      "OpenVinoBackend requires the original model shape [1, 3, H, W]");
  }

  // 宿主输入固定为 U8 NHWC BGR。颜色、归一化、布局与 FP16/FP32 转换都编入模型图，
  // 避免 CPU 每帧创建约 4.7 MiB 的 float NCHW 缓冲区。
  ov::preprocess::PrePostProcessor prepost(model);
  prepost.input().tensor()
    .set_element_type(ov::element::u8)
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  auto& preprocessing = prepost.input().preprocess();
  preprocessing.convert_element_type(ov::element::f32);
  if (config.model_color_order == ModelColorOrder::Rgb) {
    preprocessing.convert_color(ov::preprocess::ColorFormat::RGB);
  }
  if (config.normalization_divisor != 1.0F) {
    preprocessing.scale({
      config.normalization_divisor,
      config.normalization_divisor,
      config.normalization_divisor});
  }
  if (original_input_type != ov::element::f32) {
    preprocessing.convert_element_type(original_input_type);
  }
  prepost.input().model().set_layout("NCHW");

  // Decoder 的公共输出固定为 float32，即使其他模型原始输出是 FP16 也不会被误读。
  for (std::size_t index = 0; index < model->outputs().size(); ++index) {
    prepost.output(index).tensor().set_element_type(ov::element::f32);
  }
  model = prepost.build();

  // build() 后的公开输入已经从模型 NCHW 改为宿主 NHWC。
  const auto input_port = model->input();
  if (input_port.get_element_type() != ov::element::u8) {
    throw std::runtime_error("OpenVinoBackend failed to expose a uint8 host input");
  }

  const ov::Shape input_shape = input_port.get_shape();
  if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] == 0
      || input_shape[2] == 0 || input_shape[3] != 3) {
    throw std::runtime_error("OpenVinoBackend failed to expose shape [1, H, W, 3]");
  }

  input_spec_.name = input_port.get_any_name();
  input_spec_.shape.assign(input_shape.begin(), input_shape.end());

  // compile_model 才会选择 CPU/GPU 并生成可执行模型；请求池负责复用 InferRequest。
  ov::CompiledModel compiled_model = impl_->core.compile_model(model, config.device);
  std::vector<std::string> output_names;
  output_names.reserve(compiled_model.outputs().size());
  for (std::size_t index = 0; index < compiled_model.outputs().size(); ++index) {
    output_names.push_back(compiled_model.output(index).get_any_name());
  }
  auto request_pool = std::make_shared<Impl::RequestPool>(std::move(compiled_model));

  // 所有可能抛错的准备完成后再发布新模型，避免留下半初始化的运行状态。
  impl_->output_names = std::move(output_names);
  impl_->request_pool = std::move(request_pool);

  ready_ = true;
#else
  (void)config;
  throw std::runtime_error(
    "OpenVINO support is not enabled; configure with xmake f --use_openvino=y first");
#endif
}

bool OpenVinoBackend::ready() const noexcept
{
#if defined(NEWVISION_HAS_OPENVINO)
  return ready_ && impl_ != nullptr && impl_->request_pool != nullptr;
#else
  return false;
#endif
}

const InferenceInputSpec& OpenVinoBackend::inputSpec() const
{
  if (!ready()) {
    throw std::logic_error("OpenVinoBackend inputSpec() called before load()");
  }
  return input_spec_;
}

InferenceResult OpenVinoBackend::infer(const InferenceInput& input)
{
  if (!ready()) {
    throw std::logic_error("OpenVinoBackend infer() called before load()");
  }
  // 提前检查 U8 NHWC shape 和长度，避免把错误大小的 host 内存交给 SDK。
  if (input.shape != input_spec_.shape || !input.isConsistent()) {
    throw std::invalid_argument("OpenVinoBackend received an input tensor with an invalid shape");
  }

#if defined(NEWVISION_HAS_OPENVINO)
  // lease 在 Result 的零拷贝 view 中继续存活，独占本帧请求及其输出缓冲区。
  auto lease = std::make_shared<Impl::ResultLease>(impl_->request_pool);
  const std::span<const std::uint8_t> input_values = input.values();
  // OpenVINO Tensor 仅临时借用 input.values()；infer() 是同步调用，返回前 input 仍然有效。
  const ov::Shape input_shape(input.shape.begin(), input.shape.end());
  ov::Tensor input_tensor(
    ov::element::u8, input_shape, const_cast<std::uint8_t*>(input_values.data()));
  lease->infer_request.set_input_tensor(input_tensor);
  lease->infer_request.infer();

  InferenceResult result;
  result.outputs.reserve(impl_->output_names.size());
  lease->output_tensors.reserve(impl_->output_names.size());
  for (std::size_t index = 0; index < impl_->output_names.size(); ++index) {
    lease->output_tensors.push_back(lease->infer_request.get_output_tensor(index));
    const ov::Tensor& output_tensor = lease->output_tensors.back();
    if (output_tensor.get_element_type() != ov::element::f32) {
      throw std::runtime_error("OpenVinoBackend produced a non-float32 output tensor");
    }
    if (!output_tensor.is_continuous()) {
      throw std::runtime_error("OpenVinoBackend produced a non-contiguous output tensor");
    }

    // 不复制模型输出；InferenceTensor 通过 lease 保持 Tensor 和 InferRequest 的生命周期。
    const float* output_data = output_tensor.data<const float>();
    InferenceTensor output;
    output.name = impl_->output_names[index];
    const ov::Shape output_shape = output_tensor.get_shape();
    output.shape.assign(output_shape.begin(), output_shape.end());
    output.setExternalView(output_data, output_tensor.get_size(), lease);
    result.outputs.push_back(std::move(output));
  }
  return result;
#else
  (void)input;
  throw std::runtime_error("OpenVINO support is not enabled");
#endif
}

}  // namespace L2Perception
