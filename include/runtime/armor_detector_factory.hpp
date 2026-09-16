#pragma once

#include "l2_perception/armor/armor_detector.hpp"
#include "runtime/auto_aim_config.hpp"

namespace runtime {

// 按配置组装 L2 装甲检测器：建后端并加载灯条模型 → 加载数字分类器 → 构造
// ArmorDetector。实机主循环和各回放工具都走这一个函数，回放看到的就是实机
// 跑的链路；工具要换模型或设备时，改一份 config 的副本再传进来。
//
// 启动阶段调用。模型缺失或输出契约不符时抛异常，由调用方决定退化还是退出。
[[nodiscard]] L2Perception::ArmorDetector makeDetector(const AutoAimConfig& config);

}  // namespace runtime
