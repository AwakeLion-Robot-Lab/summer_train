#pragma once

#include "l2_perception/armor/armor_decoder.hpp"
#include "l2_perception/armor/light_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"

#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

/**
 * @brief detectFrame 一帧里各段的 CPU 墙钟耗时，单位 ms
 *
 * 整块 L2 常年占掉管线九成，只报一个总数没法定位是网络还是侧边灯条那一路贵，
 * 所以拆开。
 */
struct DetectTiming
{
  /// letterbox 预处理
  double preprocess{0.0};

  /// 网络推理
  double infer{0.0};

  /// 输出解码（含 NMS）
  double decode{0.0};

  /// 侧边灯条整条路：剖面搜索 + 判色 + 与已检出板的判重。
  double side_light{0.0};
};

/**
 * @brief 一帧检测里各环节的配置
 *
 * 分开传是一串位置参数，中间几个还常常要写 {} 占位才能传到后面的，收成一个
 * 结构体后调用点按名字赋值，加新环节也不动签名。
 */
struct ArmorDetectorConfig
{
  /// 整板模型的输出契约与筛选阈值
  ArmorDecoderConfig decoder{};

  /// 侧边灯条剖面搜索的门限
  LightFinderConfig finder{};

  /// letterbox 预处理参数
  ImagePreprocessConfig preprocess{};
};

/**
 * @brief L2 装甲检测的编排层
 *
 * 一帧分两部分：
 *   装甲板    整板网络（同济 yolov5 等）→ 解码，类别就是网络的 argmax；
 *   侧边灯条  只在 L3 给出预测灯条位置（LightHint）时，沿预测做剖面搜索
 *             （searchLights），剔掉属于已检出装甲板的，交给 L3 做额外的端点观测。
 *
 * 整板网络召回高，但同时看到两块板时侧面那块常常检不出，侧边灯条补的就是这块
 * 信息；它们没有类别证据，关联和门限在 L3。PnP、跟踪、开火策略都不在这里。
 */
class ArmorDetector
{
public:
  /**
   * @brief 默认构造表示"未配置模型"
   *
   * ready() 为 false，detect() 返回空结果，runtime 可以先把相机和串口跑起来。
   */
  ArmorDetector() = default;

  /**
   * @brief 用已加载模型的后端构造检测器
   *
   * 构造时按 config.decoder 核对一次输出形状，不符就抛异常，免得每帧解出垃圾
   * 角点。侧边灯条不走网络，没有第二个后端。
   *
   * @param armor_backend 必须已加载整板模型
   * @param config 解码、剖面搜索和预处理的配置
   */
  ArmorDetector(
    std::unique_ptr<IInferenceBackend> armor_backend, ArmorDetectorConfig config);

  /**
   * @brief 检测器是否可用
   * @return 有后端且后端已加载模型时返回 true
   */
  bool ready() const noexcept;

  /**
   * @brief 整图检测一帧，只要装甲板
   * @param image 相机输出的 BGR 图像
   * @return 检出的装甲板，等价于 detectFrame(image).armors
   */
  [[nodiscard]] std::vector<Armor> detect(const cv::Mat& image) const;

  /**
   * @brief 完整的一帧检测，L3 的 IESKF 走这个入口
   *
   * 后端抛出的异常在这里转成一条日志和空结果，不会中断主循环。
   *
   * @param image 相机输出的 BGR 图像
   * @param net_roi 整板网络只在这块区域里跑，缺省为整图
   * @param color 侧边灯条只保留该颜色，Unknown 表示红蓝都要。装甲板不按颜色
   *              筛，由调用方按电控给的敌方颜色过滤
   * @param hints L3 预测的侧边灯条位置，只在这些位置附近做剖面搜索；为空时
   *              不找侧边灯条，lights 为空
   * @return 本帧的装甲板和侧边灯条
   */
  [[nodiscard]] ArmorFrame detectFrame(
    const cv::Mat& image, const std::optional<cv::Rect>& net_roi = std::nullopt,
    ArmorColor color = ArmorColor::Unknown, const std::vector<LightHint>& hints = {}) const;

  /**
   * @brief 整板网络输入的宽高比
   *
   * L3 的 netFocusRoi 用它把 ROI 修成同一比例，减少 letterbox 填充。
   *
   * @return 宽 / 高，取自后端的输入形状；后端不可用时返回 1.0
   */
  [[nodiscard]] double net_aspect_ratio() const noexcept;

  /**
   * @brief 最近一帧的分段耗时，每次 detectFrame 进来先清空
   * @return 分段耗时
   */
  const DetectTiming& lastTiming() const noexcept { return last_timing_; }

  /**
   * @brief 最近一帧颜色过滤后、剔除已检出装甲板之前的全部侧边灯条候选
   * @return 侧边灯条候选，每次 detectFrame 进来先清空
   */
  const std::vector<Light>& lastLights() const noexcept { return last_lights_; }

  /**
   * @brief 当前使用的解码配置
   * @return 解码配置
   */
  const ArmorDecoderConfig& decoderConfig() const noexcept { return decoder_.config(); }

private:
  /**
   * @brief 沿预测灯条做剖面搜索，丢掉判不出颜色或颜色不符的灯条，并重排下标
   * @param image 原图
   * @param hints 预测的灯条位置
   * @param color 只保留该颜色，Unknown 表示红蓝都要
   * @return 侧边灯条候选，id 为在返回数组里的下标
   */
  std::vector<Light> findSideLights(
    const cv::Mat& image, const std::vector<LightHint>& hints, ArmorColor color) const;

  /// 整板模型后端，默认构造时为空
  std::unique_ptr<IInferenceBackend> armor_backend_;

  /// 整板模型输出解码器
  ArmorDecoder decoder_{};

  /// 剖面搜索门限
  LightFinderConfig finder_config_{};

  /// letterbox 预处理参数
  ImagePreprocessConfig preprocess_config_{};

  /// detectFrame 对外是 const，这几个快照只供调试读取，所以用 mutable。
  mutable DetectTiming last_timing_{};

  /// 最近一帧的侧边灯条候选，见 lastLights()
  mutable std::vector<Light> last_lights_;
};

/**
 * @brief 侧边灯条是否属于某块已检出的装甲板
 *
 * 灯条中心落在板四角外接框、四周各外扩 margin_by_length 倍灯条长度的范围内
 * 就算。外扩按灯条长度给，远近通用；取值要小于同一辆车相邻两块板灯条的像面
 * 间距（3 m 处约 4 倍灯长）。
 *
 * @param light 侧边灯条
 * @param armor 已检出的装甲板
 * @param margin_by_length 外扩量，以灯条长度为单位
 * @return 属于该板时返回 true
 */
[[nodiscard]] bool insideArmor(
  const Light& light, const Armor& armor, float margin_by_length);

}  // namespace L2Perception
