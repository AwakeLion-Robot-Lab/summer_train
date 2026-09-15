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
  // softmax 最大概率低于它就不认，rm_auto_aim 默认 0.7。
  double min_confidence{0.7};
};

enum class NumberVerdict
{
  Accepted,
  Negative,       // 分类为 negative：两灯条之间不是数字，典型是相邻两块板的灯条配错
  LowConfidence,  // 最大概率低于 min_confidence
  TypeMismatch    // 数字对应的板型与两灯条间距推出的大小板矛盾
};

struct NumberResult
{
  ArmorClass armor_class{ArmorClass::Unknown};
  int label_index{-1};
  double confidence{0.0};
  NumberVerdict verdict{NumberVerdict::Negative};
  // 送进 MLP 的 20x28 二值图，留给离线工具看。
  cv::Mat number_image;
};

// rm_auto_aim 的数字分类器（mlp.onnx）：两灯条之间透视抠出数字 → OTSU 二值化 →
// MLP。只回答「这是哪辆车」，不改角点。
class NumberClassifier
{
public:
  NumberClassifier() = default;

  // 启动阶段调用：读模型和标签并跑一次空输入核对输出维度。失败抛出带原因的异常。
  void load(const NumberClassifierConfig& config);
  bool ready() const noexcept { return ready_; }

  [[nodiscard]] NumberResult classify(
    const cv::Mat& bgr, const Light& left, const Light& right, bool large) const;

  // label.txt 里的原始标签，下标越界返回 "?"。
  const std::string& label(int index) const;

private:
  NumberClassifierConfig config_;
  // cv::dnn::Net::forward 不是 const，而 classify 对外是只读操作。主循环单线程，
  // 与 ArmorDetector 的调试快照同理用 mutable。
  mutable cv::dnn::Net net_;
  std::vector<std::string> labels_;
  std::vector<ArmorClass> classes_;  // 与 labels_ 一一对应，negative 为 Unknown
  bool ready_{false};
};

}  // namespace L2Perception
