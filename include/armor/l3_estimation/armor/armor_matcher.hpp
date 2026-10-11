#pragma once

#include "l3_estimation/armor/armor_observation.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/types.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <utility>
#include <vector>

// 观测关联：把本帧的检测挂到整车的某一块物理板、某一根物理灯条上。
//
// 这一步独立于滤波器，也不改目标的任何状态——两个函数都只读 target。分出来是
// 因为"这块板是车上的第几块"和"滤波器怎么吸收它"是两个可以分别验证的问题：
// 关联错了，滤波器再对也没用；关联对了，滤波器的问题才看得见。
//
// 两个函数都要一份已经外推到本帧曝光时刻的状态（EskfTarget::stateAt）。一帧
// 里只外推一次、两处共用：Motion 在 dt=0 时也会跑 clamp，反复外推不是恒等。
namespace L3Estimation {

// matchLight 各道门毙掉了多少根侧边灯条，累计值。门限只看最终采纳数是调不
// 动的：采纳数为零时，不知道是候选板槽位根本没开出来，还是某一道门收太紧。
struct LightMatchStats
{
  // 整帧没进关联：开关关着、目标是基地、本帧没关联上完整板，或 jumped 未满足。
  std::size_t frames_skipped{0};
  // 进了关联但一个候选灯条槽位都没开出来：能看见的板本帧都已配成完整板，
  // 或邻板背对相机。这时侧边灯条本来就无处可去，不算被门毙掉。
  std::size_t frames_no_candidate{0};
  // 开出来的候选灯条槽位总数，每帧 0~4 个。除以"进了关联且有槽位的帧数"就是
  // 每帧平均有几个位置能接侧边灯条，也就是采纳数的天花板——一个槽位最多收一
  // 根。参与率低的时候先看它：槽位本来就只有一个的话，再松门限也多不出来。
  std::size_t slots{0};
  // 逐 (灯条, 候选槽位) 对的计数，下面几项按门的先后顺序互斥累加。
  std::size_t considered{0};
  std::size_t reject_length{0};
  std::size_t reject_angle{0};
  std::size_t reject_chi2{0};
  std::size_t passed{0};
  // 过了三道门的 (灯条, 槽位) 对，log(实测长度 / 预测长度) 的累加。除以 passed
  // 再取 exp 是长度比的几何平均：偏离 1 说明端点定义与观测模型的灯条长度不一致，
  // 这种偏差是系统性的，会一直把滤波器往一个方向拽。
  double log_length_ratio{0.0};
  // 贪心配对之后真正返回的根数，必然不大于 passed。
  std::size_t matched{0};
  // 至少采纳了一根侧边灯条的帧数。
  std::size_t frames_matched{0};
};

// 把本帧的候选板关联到整车的各块物理板上：对每个 (观测, 板编号) 组合，把该板
// 按 state 投影出四个角点，与观测角点比中心、角度和边长，加权成一个代价，再用
// greedyMatch 在门限内贪心配对。返回 (物理板编号, 观测) 对。
//
// 候选板先按正对相机的程度排序、只留最正对的三块：四板车最多同时看到两块半，
// 取三留了余量。这只是几何近似，判的是板朝不朝着你，不判它有没有被车身挡住。
std::vector<MatchedArmor> matchArmor(
  const EskfTarget & target, const Eigen::VectorXd & state, const ObsContext & ctx,
  const std::vector<Armor> & armors);

// 侧边灯条可能出现的槽位 (板编号, 是否左灯)：最正对的那块板的左右灯条，加上
// 相邻两块板靠近它的各一根；matched_armors 里已配成完整板的板和背对相机的板
// 不算。matchLight 用它开候选，剖面搜索用它（matched_armors 传空）定搜索位置，
// 两边必须是同一组，否则搜到的灯条在关联时没有槽位可去。
std::vector<std::pair<int, bool>> lightSlots(
  const EskfTarget & target, const Eigen::VectorXd & state, const ObsContext & ctx,
  const std::vector<MatchedArmor> & matched_armors);

// 把侧边灯条关联到整车的某根物理灯条上。候选只取最正对的那块板及其两块邻板
// 靠近它的那根灯条，已配成完整板的板和背对相机的板不参与；按长度比、角度差、
// 卡方三道门筛，通过的按马氏距离贪心配对，记下 (板编号, 左右)。
//
// 本帧一块完整板都没关联上、或 require_jumped 时还没见过别的板，直接返回空：
// 没有完整板做锚，侧边灯条的编号和左右归属几乎是猜的。
//
// 卡方门限读的是 target 滤波器当前的先验协方差，调用前应已 predictEkf 到本帧。
//
// stats 非空时逐道门累加拒绝数，供 track_diag 打印；不影响关联结果。
std::vector<MatchedLight> matchLight(
  const EskfTarget & target, const Eigen::VectorXd & state, const ObsContext & ctx,
  const std::vector<L2Perception::Light> & lights,
  const std::vector<MatchedArmor> & matched_armors, LightMatchStats * stats = nullptr);

}  // namespace L3Estimation
