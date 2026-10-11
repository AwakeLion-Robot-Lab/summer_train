#pragma once

#include "l3_estimation/armor/light_measure.hpp"

#include <Eigen/Core>

#include <optional>
#include <vector>

// 一次多观测更新之后的诊断量：归一化创新平方（NIS）和端点创新的物理通道分解。
//
// 这些数字不参与滤波，只用来回答"这一帧是哪个环节偏了"。单独成文件是因为它们
// 和滤波本身是两件事：读 EskfTarget::update 想看的是"滤波器吃了什么"，中间夹
// 六十行统计代码只会挡路。
namespace L3Estimation {

// 一根灯条在像素平面上的朝向和长度，取自本帧**检测到**的端点而不是预测端点。
// 残差要投回灯条自身的坐标系，投影基必须来自观测侧，否则预测偏了的时候连基
// 都跟着偏。
struct LightAxis
{
  // 由上端点指向下端点的单位方向。
  Eigen::Vector2d direction{0.0, 1.0};
  // 像素长度，把倾角残差从像素化成弧度要除它。
  double length{0.0};
};

// 端点创新量（先验点上）投到每根灯条自己的坐标系里，分方向单位 px（倾角为
// rad）。
//
// 读法：沿灯条分量大，多半是深度（灯条长度）或高度偏了；垂直分量大，是横向
// 位置或灯条倾角（姿态）偏了。
struct LightResidual
{
  // 一个通道的统计量。mean 有符号，看的是系统偏差；rms 看的是噪声。两者
  // 必须分开：只看 RMS 分不出「检测器有偏」和「检测器抖」，而这两种病的
  // 治法完全不同——前者要在检测侧修，后者才该动 R。
  struct Channel
  {
    double mean{0.0};
    double rms{0.0};
  };

  // 把每根灯条两个端点的残差 r_top / r_bot 投到灯条方向 e 和法向 n 上，
  // 再折成四个互相正交的物理通道。L 为该灯条的像素长度：
  //   shift_perp  = (r_top·n + r_bot·n) / 2   整根灯条横向平移，px
  //   shift_along = (r_top·e + r_bot·e) / 2   整根灯条沿自身平移，px
  //   tilt        = (r_top·n − r_bot·n) / L   灯条倾角，rad
  //   length      = (r_bot·e − r_top·e)       灯条长度，px
  // 拆成这四维是因为它们互相正交、各自有物理意义：哪一维偏了直接对应
  // 哪个环节有问题。端点级的平方和把符号吃掉，看不出偏差。
  Channel shift_perp{};
  Channel shift_along{};
  Channel tilt{};
  Channel length{};

  // 端点级的 RMS，单位 px。等价于上面四个通道的重新组合（沿灯条方向有
  // along_rms² = shift_along.rms² + length.rms²/4），保留是因为历史 A/B
  // 记录用的就是这两个数。
  double along_rms_px{0.0};
  double perp_rms_px{0.0};
  // 单板深度差观测的残差，单位米。本帧没有该观测时为 0。
  double depth_diff_m{0.0};
  // 参与本次更新的灯条根数（完整板拆出的 + 独立的）。
  int light_count{0};
};

// 马氏距离平方 rᵀS⁻¹r。维数对不上或 S 不正定时返回空，让调用方自己决定是跳过
// 这一帧还是拒掉这一对，而不是拿一个退化的数去比门限。
//
// 用在两处，含义不同但算式相同：更新后取先验创新就是 NIS，关联时取单根灯条的
// 创新就是卡方门限的检验量。
std::optional<double> chi2(
  const Eigen::VectorXd & residual, const Eigen::MatrixXd & covariance);

// 把一次更新的扁平创新量拆成四个物理通道。
//
// 观测块的排布固定为 [灯条 4×n][深度差 0 或 1]，lights 与前 n 块一一对应；
// 长度对不上（说明调用方排错了块）时返回全零的结果，不去猜下标。
LightResidual analyzeLight(
  const Eigen::VectorXd & innovation, const std::vector<LightAxis> & lights,
  bool has_depth_diff);

}  // namespace L3Estimation
