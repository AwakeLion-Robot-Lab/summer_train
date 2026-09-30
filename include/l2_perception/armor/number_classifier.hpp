#pragma once

#include "l2_perception/armor.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

namespace L2Perception
{

// 分类器没有采信这块板时怎么处理。
enum class RejectPolicy
{
  Drop,  // 直接丢掉，不进 L3（sp_vision 的做法）
  Keep   // 留下，类别仍用整板网络的 argmax
};

struct NumberClassifierConfig
{
  // 关掉之后类别直接用整板网络的 argmax，不抠数字也不加载模型。默认关：
  // 见 config/auto_aim.yaml 里 number_classifier 的说明。
  bool enable{false};
  // 模型认得场上所有图案时 Drop 更稳，判不准的板不会污染关联；遇到它没见过
  // 的图案（例如别家的哨兵板）Drop 会把整辆车的观测一起筛掉，那种场合用 Keep。
  RejectPolicy on_reject{RejectPolicy::Drop};
  std::filesystem::path model_path{"model/light_model/mlp.onnx"};
  std::filesystem::path label_path{"model/light_model/label.txt"};
  // softmax 最大概率低于它就判 LowConfidence。
  double min_confidence{0.7};
  // 两灯条中心距 / 灯条长度 超过它按大板抠图。小板 135/56、大板 230/56，
  // 阈值取 rm_auto_aim 配对门限里小板和大板的分界。
  double large_center_distance_ratio{3.2};
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
  // 抠图时按几何判出的板型：两灯条中心距超过 large_center_distance_ratio
  // 倍灯条长度即为大板。它决定透视画布的宽度，也是 TypeMismatch 的比较对象。
  bool large{false};
  // 送进 MLP 的那张 20x28 二值图，留给离线工具显示。
  cv::Mat number_image;
};

// 一帧里数字二次分类的判定统计，四项之和等于送进分类器的板数。被拒的板不
// 进入后续流程，所以这几项就是"网络检出但没被采信"的明细。
struct NumberStats
{
  int accepted{0};
  int negative{0};
  int low_confidence{0};
  int type_mismatch{0};

  int dropped() const noexcept { return negative + low_confidence + type_mismatch; }
};

// 数字分类器（rm_auto_aim 的 mlp.onnx）：装甲板两侧灯条之间透视抠出数字 →
// 灰度 → OTSU 二值化 → MLP → softmax。只回答「这是哪辆车」，不碰角点。
//
// 整板网络自己也给类别，但那是在 416 像素的整图上判的，同一块板的编号会在
// 相邻帧之间跳；跟踪按编号筛观测，一跳关联就断。二次分类在原图分辨率的板
// ROI 上重判一次，并把判不准的板直接丢掉，sp_vision 的 yolov8 通路也是这么做的。
class NumberClassifier
{
public:
  NumberClassifier() = default;

  // 启动阶段调用一次：读模型、读标签，再跑一次空输入核对输出维度与标签数
  // 是否一致。任一步失败都抛出带原因的异常。
  void load(const NumberClassifierConfig& config);
  bool ready() const noexcept { return ready_; }

  // 按装甲板四角点抠图并分类。角点顺序与 Armor::corners 一致（左上、右上、
  // 右下、左下），左右灯条端点由它取出，板型由两灯条中心距推出。
  // 未 load() 时返回默认结果（verdict 为 Negative）。
  [[nodiscard]] NumberResult classify(
    const cv::Mat& bgr, const std::array<cv::Point2f, 4>& corners) const;

  // label.txt 里的原始标签，下标越界时返回 "?"。
  const std::string& label(int index) const;
  const NumberClassifierConfig& config() const noexcept { return config_; }

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
