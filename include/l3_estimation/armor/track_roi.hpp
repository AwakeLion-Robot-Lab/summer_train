#pragma once

#include "l3_estimation/armor/armor_observation.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/types.hpp"

#include <opencv2/core/types.hpp>

#include <optional>

// 由整车预测算出下一帧的搜索窗口。
//
// 和跟踪生命周期是两件事：要不要算 ROI 由跟踪器按状态机决定，算出来是什么形状
// 只取决于几何和"多久没更新了"，所以放在这里。
namespace L3Estimation::Roi {

// 一次 ROI 计算的输入。调用方（跟踪器）已经判过状态机，这里不再判。
struct Focus
{
  const EskfTarget * target{nullptr};
  ObsContext ctx{};
  // 运动外推的截止时刻，与滤波器的 hold_from 取同一个：超时之后预测停在原地，
  // ROI 也就跟着停住，不会被跑飞的速度项拖出画面。
  TimePoint motion_end{};
  // 距上次成功更新的时长和该目标的超时门限，单位秒。
  double lost_time{0.0};
  double lost_thres{0.0};
};

// 整车全部预测灯条端点的像素包围盒。一个有效端点都没有、或包围盒与画面完全
// 不相交时返回空。返回的是未裁剪的原始包围盒，裁剪留给下面两个函数。
std::optional<cv::Rect> bounds(const Focus & focus, const cv::Size & image_size);

// 独立灯条的搜索 ROI：包围框放大 1.6 倍再裁回画面。
//
// 这个 ROI 越紧越好：范围一大，别的车和环境灯光就容易混进候选，CPU 开销和
// 误匹配概率一起上去。
cv::Rect light(const cv::Rect & box, const cv::Size & image_size);

// 送给网络的检测 ROI。同样从预测包围框出发，但比 light() 多三步：按
// target_wh_ratio（网络输入宽高比）修正形状以减少 letterbox 填充、随距上次
// 更新的时长线性膨胀、再扩成方形。超时直接退化成整图。
cv::Rect net(
  const Focus & focus, const cv::Rect & box, const cv::Size & image_size,
  double target_wh_ratio);

}  // namespace L3Estimation::Roi
