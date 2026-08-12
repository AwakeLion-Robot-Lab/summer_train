#pragma once

#include <Eigen/Geometry>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace L1Sensor {

// Daedalus 单板真值旁路通道的消费者。
//
// 仿真的 `ShmMetaRegion::ground_truth` 只给到整车真值（位置 / yaw / v_yaw），
// 标定 PnP 的观测噪声 R 需要的是**每一块装甲板**的真实位姿。仿真把这些写在
// 另一段共享内存里（`/tmp/talos_ipc_plate_truth`），`ShmMetaRegion` 保持逐字节
// 不变，所以 talos-cpp 等其他消费者不受影响。
//
// 两条通道靠 `frame_seq` 对齐：真值与同序号的图像、位姿是**同一时刻**的仿真
// 世界状态，不存在插值误差 —— 这是真车拿不到的东西，也是这条通道存在的理由。
struct DaedalusGroundTruthOptions {
  std::string path{"/tmp/talos_ipc_plate_truth"};
  std::chrono::milliseconds producer_timeout{1000};
};

// 仅表示装甲板的物理尺寸。这是 L1，不能引用 L3Estimation::ArmorType。
enum class DaedalusArmorSize : std::uint8_t { Small = 0, Large = 1 };

// 一台车的整车真值，与 EKF 的 [xc, yc, z, yaw, v_yaw] 直接对应。
struct DaedalusTargetTruth {
  // 0 = red，1 = blue；-1 表示仿真给了无法识别的值。
  int team{-1};
  int armor_label{-1};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double v_yaw{0.0};
};

// 一块物理装甲板的真值。所有量都在 Daedalus 的 odom 系（z 朝上），
// 与 `DaedalusPose::position` 同一个系，长度单位 meter，角度单位 radian。
struct DaedalusPlateTruth {
  // 索引到同一快照的 targets。-1 是协议保留的“未关联”取值；仿真关联不上整车的
  // 装甲板（前哨站、基地）会被整块跳过，所以实际收到的板都是已关联的。
  int target_index{-1};
  int team{-1};
  int armor_label{-1};
  DaedalusArmorSize armor_size{DaedalusArmorSize::Small};
  // 绕车心的物理板编号，0 号是与整车 yaw 夹角最小的那块；-1 表示未关联。
  // 用它把观测残差按板分组，才能分别核对 r1/r2 和 z2-z1。
  int plate_index{-1};

  // 板中心，即四个角点的均值。
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  // 板系相对世界系的姿态。右手系：x 沿板法线**朝外**，z 朝上，
  // y = z × x 指向观察者的右手边（与枪管系的 y 相反，因为这里的 x 迎着观察者）。
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  // 四个角点，顺序固定为左上、右上、右下、左下，与 L2/L3 的 Armor 一致。
  std::array<Eigen::Vector3d, 4> corners{};

  // 板法线的水平朝向，法线**朝外**（背离车心）。
  //
  // 注意这与 newvision 的板 yaw 差 π：`L3Estimation::Armor::ypr_in_world[0]` 用的是
  // 板系 x 轴，而 `pnp_solver.cpp` 的 armorPoints 把板系定成 y=观察者左、z=上，
  // 于是 x = y × z 指向**车内**。要和 EKF 的四维观测比对，必须先过
  // `toNewvisionArmorYaw()`。
  double yaw{0.0};
  // 板法线的仰角；装甲板后仰 15° 时为负。
  double pitch{0.0};
  // 板中心到整车中心的水平距离，即 EKF 的 r；未关联到整车时为 0。
  double radius{0.0};
};

// 一帧的完整真值快照。整车真值在这里重复了一份，所以一次读取拿到的就是自洽的
// 一帧，不必跨两段共享内存去凑。
struct DaedalusGroundTruth {
  std::uint64_t frame_seq{0};
  // 与 `DaedalusFrame::timestamp_ns` 同一时钟域（system_clock）。跨进程配对请用
  // frame_seq，时间戳只用于记录。
  std::uint64_t timestamp_ns{0};
  std::vector<DaedalusTargetTruth> targets;
  std::vector<DaedalusPlateTruth> plates;
};

// 把 `DaedalusPlateTruth::yaw`（朝外法线）换算成 newvision 的板 yaw 约定
// （朝内法线），结果收敛到 (-π, π]。忘掉这一步的话每个 yaw 残差都会差 π，
// 而且看起来像是"检测全错"而不是"约定不一致"。
[[nodiscard]] double toNewvisionArmorYaw(double outward_yaw) noexcept;

class DaedalusGroundTruthSource {
public:
  [[nodiscard]] static std::unique_ptr<DaedalusGroundTruthSource> connect(
    const DaedalusGroundTruthOptions& options = {},
    std::string* error = nullptr);

  ~DaedalusGroundTruthSource();

  DaedalusGroundTruthSource(const DaedalusGroundTruthSource&) = delete;
  DaedalusGroundTruthSource& operator=(const DaedalusGroundTruthSource&) = delete;

  // 按帧号取真值。环形槽位只保留最近 slotCount() 帧，消费者落后太多就取不到；
  // 取不到时返回 nullopt 而不是相邻帧 —— 拿错帧的真值比没有真值更糟。
  [[nodiscard]] std::optional<DaedalusGroundTruth> forFrame(
    std::uint64_t frame_seq) const;

  // 最近一次发布的真值，用于不需要和图像配对的场合。
  [[nodiscard]] std::optional<DaedalusGroundTruth> latest() const;

  [[nodiscard]] bool producerAlive() const noexcept;

  // 环形槽位数，即 forFrame() 能回溯的最大帧数。
  [[nodiscard]] std::size_t slotCount() const noexcept;

  // 取走最近一次协议或数据错误；没有错误时返回空字符串。
  [[nodiscard]] std::string takeLastError();

private:
  struct Impl;
  explicit DaedalusGroundTruthSource(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace L1Sensor
