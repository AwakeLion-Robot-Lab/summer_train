#pragma once

#include "l2_perception/armor/armor_detector.hpp"
#include "runtime/auto_aim_config.hpp"

namespace runtime {

// 按 auto_aim.yaml 组装 L2 装甲检测器：灯条关键点模型 + 数字分类。实机主循环和
// 各回放工具共用这一份，回放看到的就是实机跑的链路；工具要换模型或设备时改
// config 的副本再传进来。
//
// 启动阶段调用。灯条模型或数字分类器缺失、输出契约不符时抛异常，由调用方决定
// 是退化还是退出。
[[nodiscard]] L2Perception::ArmorDetector makeDetector(const AutoAimConfig& config);

}  // namespace runtime
