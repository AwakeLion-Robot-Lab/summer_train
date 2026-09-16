#pragma once

#include "l2_perception/armor.hpp"

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

namespace L2Perception
{

struct NumberClassifierConfig
{
  std::filesystem::path model_path{"model/light_model/mlp.onnx"};
  std::filesystem::path label_path{"model/light_model/label.txt"};
  // softmax 最大概率低于它就判 LowConfidence。
  double min_confidence{0.7};
};

enum class NumberVerdict
{
  Accepted,
  Negative,       // 分到 negative 类：两灯条之间不是数字，典型是配错的一对灯条
  LowConfidence,  // 最大概率低于 min_confidence
  TypeMismatch    // 数字对应的板型与两灯条间距推出的大小板矛盾
};

struct NumberResult
{
  ArmorClass armor_class{ArmorClass::Unknown};
  int label_index{-1};
  double confidence{0.0};
  NumberVerdict verdict{NumberVerdict::Negative};
  // 送进 MLP 的那张 20x28 二值图，留给离线工具显示。
  cv::Mat number_image;
};

// 数字分类器（mlp.onnx）：两灯条之间透视抠出数字 → 灰度 → OTSU 二值化 →
// MLP → softmax。只回答「这是哪辆车」，不碰角点。
class NumberClassifier
{
public:
  NumberClassifier() = default;

  // 启动阶段调用一次：读模型、读标签，再跑一次空输入核对输出维度与标签数
  // 是否一致。任一步失败都抛出带原因的异常。
  void load(const NumberClassifierConfig& config);
  bool ready() const noexcept { return ready_; }

  // 对一对灯条抠图并分类。large 决定透视目标画布的宽度（大板更宽），
  // 未 load() 时返回默认结果（verdict 为 Negative）。
  [[nodiscard]] NumberResult classify(
    const cv::Mat& bgr, const Light& left, const Light& right, bool large) const;

  // label.txt 里的原始标签，下标越界时返回 "?"。
  const std::string& label(int index) const;

private:
  NumberClassifierConfig config_;
  // cv::dnn::Net::forward 不是 const，而 classify 对外是只读的；主循环单线程，
  // 所以用 mutable。
  mutable cv::dnn::Net net_;
  std::vector<std::string> labels_;
  std::vector<ArmorClass> classes_;  // 与 labels_ 逐行对应，negative 记为 Unknown
  bool ready_{false};
};

}  // namespace L2Perception
