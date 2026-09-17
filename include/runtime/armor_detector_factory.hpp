#pragma once

#include "l2_perception/armor/armor_detector.hpp"
#include "runtime/auto_aim_config.hpp"

namespace runtime {

// 按配置组装 L2 装甲检测器：建后端并加载整板模型 → 需要时加载侧边灯条的
// 关键点模型 → 构造 ArmorDetector。实机主循环和各回放工具都走这一个函数，
// 回放看到的就是实机跑的链路；工具要换模型或设备时，改一份 config 的副本再
// 传进来。
//
// guess_layout 为 true 时按模型输出名认 decoder 预设（阈值也取预设的），给
// 命令行临时换模型的离线工具用；实机保持 false，契约只认 inference.decoder。
//
// 启动阶段调用。模型缺失或输出契约不符时抛异常，由调用方决定退化还是退出。
[[nodiscard]] L2Perception::ArmorDetector makeDetector(
  const AutoAimConfig& config, bool guess_layout = false);

}  // namespace runtime
