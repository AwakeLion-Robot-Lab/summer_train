#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>

/**
 * @file
 * @brief L3 里与目标类型无关的公共定义
 *
 * 装甲板专有的类别、观测和配置在 armor/types.hpp，符的在 buff/ 下，顶层不认识
 * 任何具体目标。
 */
namespace L3Estimation {

/// 时间戳统一用单调时钟，系统时间被校准时也不会出现负的帧间隔。
using TimePoint = std::chrono::steady_clock::time_point;

/**
 * @brief from 到 to 的秒数，钳到非负
 *
 * 避免乱序时间戳算出负的 dt。
 *
 * @param from 起始时刻
 * @param to 结束时刻
 * @return max(0, to - from)，单位 s
 */
inline double elapsedSeconds(const TimePoint & from, const TimePoint & to) noexcept
{
  return std::max(0.0, std::chrono::duration<double>(to - from).count());
}

/**
 * @brief 四态跟踪状态机，装甲板和符共用
 *
 * 转移规则见 tracking/lifecycle.hpp。
 */
enum class TrackState : std::uint8_t {
  Lost,       ///< 当前没有可用目标
  Detecting,  ///< 已发现目标，等待连续帧确认
  Tracking,   ///< 稳定跟踪
  TempLost    ///< 短时丢失，继续输出预测状态
};

/**
 * @brief 跟踪状态的名字
 *
 * 日志、叠加层和回放 CSV 共用这一份名字，各处自己写会拼法漂移。
 *
 * @param state 跟踪状态
 * @return 状态名
 */
std::string toString(TrackState state);

}  // namespace L3Estimation
