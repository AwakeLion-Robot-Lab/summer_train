#pragma once

#include <chrono>
#include <cstdint>

// L3 里与目标类型无关的公共定义。装甲板专有的类别、观测和配置在
// armor/types.hpp，符的在 buff/ 下，顶层不认识任何具体目标。
namespace L3Estimation {

// 时间戳统一用单调时钟，系统时间被校准时也不会出现负的帧间隔。
using TimePoint = std::chrono::steady_clock::time_point;

// 四态跟踪状态机，装甲板和符共用。转移规则见 association.hpp 的 updateFsm。
enum class TrackState : std::uint8_t {
  Lost,       // 当前没有可用目标
  Detecting,  // 已发现目标，等待连续帧确认
  Tracking,   // 稳定跟踪
  TempLost    // 短时丢失，继续输出预测状态
};

}  // namespace L3Estimation
