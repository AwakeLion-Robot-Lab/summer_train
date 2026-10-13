#pragma once

#include "l2_perception/armor/armor_detector.hpp"
#include "runtime/auto_aim_config.hpp"

namespace runtime {

/**
 * @brief 按配置组装 L2 装甲检测器
 *
 * 建后端并加载整板模型 → 需要时按输出形状认 decoder 预设 → 构造 ArmorDetector。
 * 侧边灯条走剖面搜索，不加载第二个模型。实机主循环和各回放工具都走这一个函数，
 * 回放看到的就是实机跑的链路；工具要换模型或设备时，改一份 config 的副本再
 * 传进来。
 *
 * OpenVINO 的非 CPU 设备（GPU）加载失败时记一条警告并退回 CPU。
 *
 * 启动阶段调用。模型缺失或输出契约不符时抛异常，由调用方决定退化还是退出。
 *
 * @param config 自瞄配置，读其中的模型、后端、decoder 与 light_finder
 * @param guess_layout 为 true 时按模型输出形状认 decoder 预设（阈值也取预设的），
 *                     给命令行临时换模型的离线工具用；实机保持 false。YAML 写
 *                     layout: auto 时同样按形状认，但阈值取 YAML 里写了的
 * @return 可用的检测器
 */
[[nodiscard]] L2Perception::ArmorDetector makeDetector(
  const AutoAimConfig& config, bool guess_layout = false);

}  // namespace runtime
