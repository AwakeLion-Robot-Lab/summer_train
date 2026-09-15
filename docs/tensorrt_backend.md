# TensorRT L2 后端

`TensorRtBackend` 与 `OpenVinoBackend` 共用 `IInferenceBackend` 接口：

- L2 输入仍是连续的 `U8 NHWC BGR`，由后端完成 RGB/BGR、归一化和 `NCHW` 转换。
- 模型输入要求静态 `[1, 3, H, W]`，类型支持 FP32/FP16。
- `model_path` 可以是 TensorRT `.engine`/`.plan`，也可以是 `.onnx`；ONNX 在 `load()` 时构建 engine。
- I/O 必须是紧密连续的 `kLINEAR` 格式。ONNX 构建时会强制该格式，外部 engine 若使用
  CHW2/HWC8 等向量化格式会在分配 CUDA buffer 前被明确拒绝，避免尺寸低估和越界写入。
- 推理结果统一为拥有自身内存的 float32 `InferenceTensor`，不会被下一帧覆盖。
- 当前实现使用 TensorRT 8.5+ 的 named I/O 和 `enqueueV3` 接口；动态 shape 暂不接受。
- `load()` 和 `infer()` 会在退出时恢复调用线程原来的 CUDA device；失败重载不会破坏
  已经可用的旧 engine。

接入 `ArmorDetector` 时无需修改 L2 的预处理和解码代码，只需替换后端对象：

```cpp
auto backend = std::make_unique<L2Perception::TensorRtBackend>();
L2Perception::InferenceModelConfig model;
model.model_path = "model/armor_model/0526.engine";
model.device = "CUDA:0";
backend->load(model);
L2Perception::ArmorDetector detector(std::move(backend));
```

实际运行时也可以在 `config/auto_aim.yaml` 或 `config/daedalus.yaml` 中选择后端：

```yaml
inference:
  backend: "tensorrt"
  model_path: "model/armor_model/0526.onnx"
  device: "CUDA:0"
```

默认仍为 `openvino`，所以未安装 TensorRT 的机器无需修改配置。

启用可选依赖：

```bash
xmake f --use_tensorrt=y \
  --tensorrt_root=/path/to/TensorRT \
  --cuda_root=/usr/local/cuda
xmake build newvision
```

`InferenceModelConfig::device` 使用 `CUDA`、`CUDA:0`、`GPU` 或 `GPU:0`。没有 CUDA/TensorRT SDK 的机器保持默认配置即可；此时后端会在 `load()` 给出明确错误，接口 smoke test 仍可运行：

```bash
xmake build tensorrt_backend_smoke
xmake run tensorrt_backend_smoke
```

有 GPU 时可把 engine/ONNX 路径作为 smoke test 第一个参数，设备作为第二个参数：

```bash
xmake run tensorrt_backend_smoke -- model/armor_model/0526.onnx CUDA:0
```
