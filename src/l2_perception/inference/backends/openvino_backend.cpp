#include "l2_perception/inference/backends/openvino_backend.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <atomic>
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
  // 整条自瞄链路是单帧同步的，所以只创建一个 InferRequest，没有请求池。
  // 输出走零拷贝：InferenceTensor 直接指向这个请求的输出缓冲区，因此在上一帧
  // 结果被释放之前不能开始下一帧推理，否则会就地改写别人正在读的检测结果。
  // busy 就是这道闸——占用中再次推理直接抛，而不是悄悄覆盖。
  struct Request
  {
    explicit Request(ov::CompiledModel model)
      : compiled_model(std::move(model))
      , infer_request(compiled_model.create_infer_request())
    {
    }

    ov::CompiledModel compiled_model;  // 必须活得比 infer_request 长。
    ov::InferRequest infer_request;
    std::atomic_bool busy{false};
  };

  std::shared_ptr<Request> request;
  std::vector<std::string> output_names;

  // 结果租约。它被塞进 InferenceTensor 的零拷贝视图里，所以只要还有人拿着
  // 结果，请求就活着——Backend 先析构也安全。
  //
  // 反过来说：**不要跨帧保存 InferenceResult**。租约没释放，下一帧 infer()
  // 会抛异常。需要留数据就自己拷一份出来。
  struct ResultLease
  {
    explicit ResultLease(std::shared_ptr<Request> request)
      : request(std::move(request))
      , infer_request(this->request->infer_request)
    {
      if (this->request->busy.exchange(true)) {
        throw std::runtime_error(
          "OpenVINO infer request is still held by a previous InferenceResult; "
          "copy what you need instead of keeping the result across frames");
      }
    }

    ~ResultLease()
    {
      // 先放掉额外的 Tensor 句柄，再解除占用。
      output_tensors.clear();
      request->busy.store(false);
    }

    std::shared_ptr<Request> request;
    ov::InferRequest& infer_request;  // 就是 request->infer_request，省一层解引用。
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
  auto& input = prepost.input();
  const ov::PartialShape host_input_shape{
    1,
    static_cast<std::int64_t>(original_input_shape[2]),
    static_cast<std::int64_t>(original_input_shape[3]),
    3};
  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape(host_input_shape)
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);
  input.model().set_layout("NCHW");

  auto& preprocessing = input.preprocess();
  preprocessing.convert_element_type(ov::element::f32);
  if (config.model_color_order == ModelColorOrder::Rgb) {
    preprocessing.convert_color(ov::preprocess::ColorFormat::RGB);
  }
  if (config.normalization_divisor != 1.0F) {
    preprocessing.scale(config.normalization_divisor);
  }

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

  // compile_model 才会选择 CPU/GPU 并生成可执行模型。
  // 编译属性只影响调度，不改变数值结果。
  ov::AnyMap compile_properties;
  if (config.latency_hint) {
    // performance_mode 是设备无关的通用 hint，CPU 和 GPU 插件都接受。
    compile_properties.emplace(
      ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY);
  }

  // 线程数、大小核调度和超线程只有 CPU 插件认识。GPU 插件收到这些会直接抛异常
  // 而不是忽略，所以必须按设备过滤；device 为 GPU 时它们无意义，静默跳过即可。
  const bool cpu_device = config.device == "CPU" || config.device.starts_with("CPU.");
  if (cpu_device) {
    if (config.inference_num_threads > 0) {
      compile_properties.emplace(
        ov::inference_num_threads.name(), static_cast<int>(config.inference_num_threads));
    }
    // Any 表示不下发这一项，保持插件默认；显式写 ANY_CORE 在只有大核的 CPU 上
    // 同样合法，但没必要多发一个属性。
    switch (config.scheduling_core_type) {
      case SchedulingCoreType::PCoreOnly:
        compile_properties.emplace(
          ov::hint::scheduling_core_type.name(), ov::hint::SchedulingCoreType::PCORE_ONLY);
        break;
      case SchedulingCoreType::ECoreOnly:
        compile_properties.emplace(
          ov::hint::scheduling_core_type.name(), ov::hint::SchedulingCoreType::ECORE_ONLY);
        break;
      case SchedulingCoreType::Any:
        break;
    }
    if (config.enable_hyper_threading) {
      compile_properties.emplace(
        ov::hint::enable_hyper_threading.name(), *config.enable_hyper_threading);
    }
  }

  ov::CompiledModel compiled_model =
    impl_->core.compile_model(model, config.device, compile_properties);
  std::vector<std::string> output_names;
  output_names.reserve(compiled_model.outputs().size());
  for (std::size_t index = 0; index < compiled_model.outputs().size(); ++index) {
    output_names.push_back(compiled_model.output(index).get_any_name());
  }
  auto request = std::make_shared<Impl::Request>(std::move(compiled_model));

  // 所有可能抛错的准备完成后再发布新模型，避免留下半初始化的运行状态。
  impl_->output_names = std::move(output_names);
  impl_->request = std::move(request);

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
  return ready_ && impl_ != nullptr && impl_->request != nullptr;
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
  auto lease = std::make_shared<Impl::ResultLease>(impl_->request);
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
