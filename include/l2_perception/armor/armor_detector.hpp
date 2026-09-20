#pragma once

#include "l2_perception/armor/armor_decoder.hpp"
#include "l2_perception/armor/armor_refiner.hpp"
#include "l2_perception/armor/light_detector.hpp"
#include "l2_perception/armor/number_classifier.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"

#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// L2 装甲检测的编排层，一帧分两部分：
//   装甲板  整板网络（同济 yolov5 等）→ 解码 → 板 ROI 内传统精修角点 →
//          数字二次分类（可关）；
//   侧边灯条  只在 L3 给出 light_roi 时，在其中用传统二值化（findLights）找
//            灯条，剔掉属于已检出装甲板的，交给 L3 做额外的端点观测。
//
// 整板网络召回高，但同时看到两块板时侧面那块常常检不出，侧边灯条补的就是这块
// 信息；它们没有类别证据，关联和门限在 L3。PnP、跟踪、开火策略都不在这里。
// 一帧检测里各环节的配置。分开传是六个位置参数，中间几个还常常要写 {} 占位
// 才能传到后面的，收成一个结构体后调用点按名字赋值，加新环节也不动签名。
// detectFrame 一帧里各段的 CPU 墙钟耗时，单位 ms。整块 L2 常年占掉管线九成，
// 只报一个总数没法定位是网络、精修还是侧边灯条那一路贵，所以拆开。
struct DetectTiming
{
  double preprocess{0.0};
  double infer{0.0};
  double decode{0.0};
  double refine{0.0};
  double number{0.0};
  // 侧边灯条整条路：二值化 + 轮廓筛选 + 判色 + 与已检出板的判重。
  double side_light{0.0};
};

struct ArmorDetectorConfig
{
  ArmorDecoderConfig decoder{};
  ArmorRefinerConfig refiner{};
  LightFinderConfig finder{};
  ImagePreprocessConfig preprocess{};
  NumberClassifierConfig number{};
};

class ArmorDetector
{
public:
  // 默认构造表示“未配置模型”：ready() 为 false，detect() 返回空结果，runtime
  // 可以先把相机和串口跑起来。
  ArmorDetector() = default;

  // armor_backend 必须已加载整板模型，构造时按 config.decoder 核对一次输出形状，
  // 不符就抛异常，免得每帧解出垃圾角点。侧边灯条不走网络，没有第二个后端。
  ArmorDetector(
    std::unique_ptr<IInferenceBackend> armor_backend, ArmorDetectorConfig config);

  bool ready() const noexcept;
  // 整图检测一帧，只要装甲板，等价于 detectFrame(image).armors。
  [[nodiscard]] std::vector<Armor> detect(const cv::Mat& image) const;

  // 完整的一帧检测，L3 的 IESKF 走这个入口。
  //   light_roi 有值时在其中找侧边灯条，无值时 lights 为空；
  //   net_roi   整板网络只在这块区域里跑，缺省为整图；
  //   color     侧边灯条只保留该颜色，Unknown 表示红蓝都要。装甲板不按颜色
  //             筛，由调用方按电控给的敌方颜色过滤。
  // 后端抛出的异常在这里转成一条日志和空结果，不会中断主循环。
  [[nodiscard]] ArmorFrame detectFrame(
    const cv::Mat& image, const std::optional<cv::Rect>& light_roi = std::nullopt,
    const std::optional<cv::Rect>& net_roi = std::nullopt,
    ArmorColor color = ArmorColor::Unknown) const;

  // 整板网络输入的宽高比（宽 / 高），取自后端的输入形状；后端不可用时返回 1.0。
  // L3 的 netFocusRoi 用它把 ROI 修成同一比例，减少 letterbox 填充。
  [[nodiscard]] double net_aspect_ratio() const noexcept;

  // 最近一帧的调试快照：精修统计、逐块明细（collectRecords(true) 之后才填，
  // 实机保持关闭以免每帧多分配），以及颜色过滤后、剔除已检出装甲板之前的
  // 全部侧边灯条候选。每次 detectFrame 进来先清空。
  const RefineStats& lastRefine() const noexcept { return last_refine_; }
  const DetectTiming& lastTiming() const noexcept { return last_timing_; }
  const NumberStats& lastNumbers() const noexcept { return last_numbers_; }
  const std::vector<RefineRecord>& lastRecords() const noexcept { return last_records_; }
  const std::vector<Light>& lastLights() const noexcept { return last_lights_; }
  void collectRecords(bool enable) noexcept { collect_records_ = enable; }

  const ArmorDecoderConfig& decoderConfig() const noexcept { return decoder_.config(); }

private:
  // 逐块重判类别：接受的改写 class_id，其余直接从 armors 里删掉。分类器没
  // 加载时原样返回，类别仍是网络的 argmax。
  NumberStats classifyNumbers(const cv::Mat& image, std::vector<Armor>& armors) const;

  std::vector<Light> findSideLights(
    const cv::Mat& image, const cv::Rect& roi, ArmorColor color) const;

  std::unique_ptr<IInferenceBackend> armor_backend_;
  ArmorDecoder decoder_{};
  ArmorRefiner refiner_{};
  NumberClassifier classifier_{};
  LightFinderConfig finder_config_{};
  ImagePreprocessConfig preprocess_config_{};
  bool collect_records_{false};
  // detectFrame 对外是 const，这几个快照只供调试读取，所以用 mutable。
  mutable RefineStats last_refine_{};
  mutable DetectTiming last_timing_{};
  mutable NumberStats last_numbers_{};
  mutable std::vector<RefineRecord> last_records_;
  mutable std::vector<Light> last_lights_;
};

// 侧边灯条是否属于某块已检出的装甲板：灯条中心落在板四角外接框、四周各外扩
// margin_by_length 倍灯条长度的范围内就算。外扩按灯条长度给，远近通用；取值
// 要小于同一辆车相邻两块板灯条的像面间距（3 m 处约 4 倍灯长）。
[[nodiscard]] bool insideArmor(
  const Light& light, const Armor& armor, float margin_by_length);

}  // namespace L2Perception
