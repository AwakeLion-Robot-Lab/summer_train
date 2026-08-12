#pragma once

namespace L3Estimation::FilterEst {

// EKF 后端独有的观测噪声与收敛策略。共享物理模型参数在
// L3Estimation::TargetConfig 中，两个结构体刻意不互相包含。
struct TargetConfig {
  // [方位角, 俯仰角, 距离, 装甲板 yaw] 的观测方差基底。
  double angle_variance{4e-3};
  double distance_variance{1.0};
  double armor_yaw_variance{9e-2};

  int min_update_count{3};
  int outpost_min_update_count{10};
  // 最近 NIS 窗口中失败比例超过该值时丢弃 EKF 目标。
  double max_nis_failure_ratio{0.4};
  // 前哨站收敛后只保留旋转方向，将转速吸附到该量级。
  double outpost_v_yaw{2.51};
};

}  // namespace L3Estimation::FilterEst
