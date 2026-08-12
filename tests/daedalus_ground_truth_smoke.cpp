#include "l1_sensor/daedalus_ground_truth.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <new>
#include <numbers>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {

// 这份布局是 crates/talos-ipc/src/plate_truth.rs 的第三份镜像（生产者 Rust、
// 消费者 daedalus_ground_truth.cpp、以及这里）。故意重写一遍而不是复用消费者的
// 私有结构：如果哪天两边的偏移对不上，这个测试要能发现，而不是跟着一起错。
constexpr std::uint32_t kMagic = 0x54505401;
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kMaxTargets = 16;
constexpr std::size_t kMaxPlates = 64;
constexpr std::size_t kSlots = 8;
constexpr std::uint32_t kUnlinked = 0xFFFFFFFFU;
constexpr std::uint8_t kUnlinkedIndex = 0xFFU;

struct alignas(64) PlateTruth {
  std::uint32_t target_index;
  std::uint8_t team;
  std::uint8_t armor_label;
  std::uint8_t armor_type;
  std::uint8_t plate_index;
  float center[3];
  float quaternion[4];
  float corners[4][3];
  float yaw;
  float pitch;
  float radius;
  std::uint8_t padding[32];
};

struct alignas(32) GroundTruthTarget {
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint8_t team;
  std::uint8_t armor_label;
  std::uint8_t is_outpost;
  std::uint8_t padding1;
  float position[3];
  float vyaw;
  float yaw;
  std::uint8_t padding[24];
};

struct alignas(64) PlateTruthBatch {
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t target_count;
  std::uint32_t plate_count;
  GroundTruthTarget targets[kMaxTargets];
  PlateTruth plates[kMaxPlates];
};

struct alignas(64) PlateTruthSlot {
  alignas(64) std::atomic<std::uint64_t> seq;
  std::uint8_t padding[56];
  PlateTruthBatch batch;
};

struct alignas(64) PlateTruthHeader {
  std::uint32_t magic;
  std::uint32_t version;
  alignas(8) std::atomic<std::uint64_t> latest_frame_seq;
  std::uint64_t created_ns;
  std::uint64_t heartbeat_ns;
  std::uint32_t max_targets;
  std::uint32_t max_plates;
  std::uint32_t slot_count;
  std::uint8_t padding[20];
};

struct PlateTruthRegion {
  PlateTruthHeader header;
  PlateTruthSlot slots[kSlots];
};

static_assert(sizeof(PlateTruth) == 128);
static_assert(sizeof(GroundTruthTarget) == 64);
static_assert(sizeof(PlateTruthBatch) == 9280);
static_assert(sizeof(PlateTruthSlot) == 9344);
static_assert(sizeof(PlateTruthHeader) == 64);
static_assert(sizeof(PlateTruthRegion) == 74816);
static_assert(offsetof(PlateTruthRegion, slots) == 64);

void check(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint64_t systemNowNs()
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

// 一段自己创建、自己 mmap 的共享内存，扮演仿真那一侧的生产者。
class FakeProducer {
public:
  explicit FakeProducer(std::string path) : path_(std::move(path))
  {
    const int fd = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    check(fd >= 0, "cannot create " + path_);
    check(
      ::ftruncate(fd, static_cast<off_t>(sizeof(PlateTruthRegion))) == 0,
      "cannot size " + path_);

    void* data = ::mmap(
      nullptr, sizeof(PlateTruthRegion), PROT_READ | PROT_WRITE, MAP_SHARED,
      fd, 0);
    ::close(fd);
    check(data != MAP_FAILED, "cannot mmap " + path_);

    region_ = ::new (data) PlateTruthRegion{};
    region_->header.magic = kMagic;
    region_->header.version = kVersion;
    region_->header.created_ns = systemNowNs();
    region_->header.heartbeat_ns = region_->header.created_ns;
    region_->header.max_targets = kMaxTargets;
    region_->header.max_plates = kMaxPlates;
    region_->header.slot_count = kSlots;
    region_->header.latest_frame_seq.store(0, std::memory_order_release);
    for (auto& slot : region_->slots) {
      slot.seq.store(0, std::memory_order_release);
    }
  }

  ~FakeProducer()
  {
    if (region_ != nullptr) {
      ::munmap(region_, sizeof(PlateTruthRegion));
    }
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  FakeProducer(const FakeProducer&) = delete;
  FakeProducer& operator=(const FakeProducer&) = delete;

  void publish(const PlateTruthBatch& batch)
  {
    auto& slot = region_->slots[batch.frame_seq % kSlots];
    slot.seq.fetch_add(1, std::memory_order_acq_rel);
    std::atomic_thread_fence(std::memory_order_release);
    std::memcpy(&slot.batch, &batch, sizeof(PlateTruthBatch));
    std::atomic_thread_fence(std::memory_order_release);
    slot.seq.fetch_add(1, std::memory_order_acq_rel);

    region_->header.heartbeat_ns = systemNowNs();
    region_->header.latest_frame_seq.store(
      batch.frame_seq, std::memory_order_release);
  }

  // 把某个槽位卡在"正在写"的状态，用来验证消费者不会读走半截数据。
  void beginTornWrite(std::uint64_t frame_seq)
  {
    region_->slots[frame_seq % kSlots].seq.fetch_add(
      1, std::memory_order_acq_rel);
  }

  void setVersion(std::uint32_t version) { region_->header.version = version; }

  void setHeartbeat(std::uint64_t heartbeat_ns)
  {
    region_->header.heartbeat_ns = heartbeat_ns;
  }

private:
  std::string path_;
  PlateTruthRegion* region_{nullptr};
};

// 一块位于 (3, 0, 0.1)、法线指向 -x（正对着原点的相机）的小装甲板。
PlateTruthBatch makeBatch(std::uint64_t frame_seq)
{
  PlateTruthBatch batch{};
  batch.frame_seq = frame_seq;
  batch.timestamp_ns = 1'000'000ULL * frame_seq;
  batch.target_count = 1;
  batch.plate_count = 1;

  batch.targets[0].frame_seq = frame_seq;
  batch.targets[0].timestamp_ns = batch.timestamp_ns;
  batch.targets[0].team = 1;
  batch.targets[0].armor_label = 3;
  batch.targets[0].position[0] = 3.2F;
  batch.targets[0].position[1] = 0.0F;
  batch.targets[0].position[2] = 0.0F;
  batch.targets[0].yaw = 0.5F;
  batch.targets[0].vyaw = -2.5F;

  PlateTruth& plate = batch.plates[0];
  plate.target_index = 0;
  plate.team = 1;
  plate.armor_label = 3;
  plate.armor_type = 0;
  plate.plate_index = 2;
  plate.center[0] = 3.0F;
  plate.center[1] = 0.0F;
  plate.center[2] = 0.1F;
  // 法线 -x、上 +z 的板系相对世界系是绕 z 转 180°。
  plate.quaternion[0] = 0.0F;
  plate.quaternion[1] = 0.0F;
  plate.quaternion[2] = 0.0F;
  plate.quaternion[3] = 1.0F;
  // 左上、右上、右下、左下 —— 从板正前方（相机在原点）看过去，
  // 相机的右是世界 -y。
  const float half_w = 0.0675F;
  const float half_h = 0.0275F;
  const float corners[4][3] = {
    {3.0F, +half_w, 0.1F + half_h},
    {3.0F, -half_w, 0.1F + half_h},
    {3.0F, -half_w, 0.1F - half_h},
    {3.0F, +half_w, 0.1F - half_h},
  };
  std::memcpy(plate.corners, corners, sizeof(corners));
  plate.yaw = static_cast<float>(std::numbers::pi);
  plate.pitch = 0.0F;
  plate.radius = 0.2F;
  return batch;
}

}  // namespace

int main()
{
  using namespace L1Sensor;

  const std::string path =
    (std::filesystem::temp_directory_path() /
     ("daedalus_ground_truth_smoke_" + std::to_string(::getpid()))).string();

  try {
    {
      DaedalusGroundTruthOptions missing;
      missing.path = path + "_absent";
      std::string error;
      check(
        DaedalusGroundTruthSource::connect(missing, &error) == nullptr,
        "connect must fail when the region does not exist");
      check(!error.empty(), "a failed connect must report a reason");
    }

    FakeProducer producer(path);

    DaedalusGroundTruthOptions options;
    options.path = path;

    {
      producer.setVersion(kVersion + 1);
      std::string error;
      check(
        DaedalusGroundTruthSource::connect(options, &error) == nullptr,
        "connect must reject a mismatched protocol version");
      producer.setVersion(kVersion);
    }

    std::string error;
    auto source = DaedalusGroundTruthSource::connect(options, &error);
    check(source != nullptr, "connect failed: " + error);
    check(source->slotCount() == kSlots, "slot count mismatch");
    check(source->producerAlive(), "a fresh heartbeat must read as alive");

    producer.publish(makeBatch(41));
    check(
      !source->forFrame(40).has_value(),
      "an unpublished frame must not resolve to a neighbouring frame");

    auto truth = source->forFrame(41);
    check(truth.has_value(), "frame 41 must resolve");
    check(truth->frame_seq == 41, "frame_seq round-trip failed");
    check(truth->targets.size() == 1, "target count round-trip failed");
    check(truth->plates.size() == 1, "plate count round-trip failed");

    const DaedalusTargetTruth& target = truth->targets.front();
    check(target.team == 1, "target team round-trip failed");
    check(
      std::abs(target.position.x() - 3.2) < 1e-6,
      "target position round-trip failed");
    check(std::abs(target.v_yaw + 2.5) < 1e-6, "target v_yaw round-trip failed");

    const DaedalusPlateTruth& plate = truth->plates.front();
    check(plate.target_index == 0, "plate must link back to its target");
    check(plate.plate_index == 2, "plate index round-trip failed");
    check(
      plate.armor_size == DaedalusArmorSize::Small, "armor size round-trip failed");
    check(
      std::abs(plate.center.x() - 3.0) < 1e-6, "plate centre round-trip failed");
    check(
      std::abs(plate.radius - 0.2) < 1e-6, "plate radius round-trip failed");

    // 角点顺序是左上、右上、右下、左下：前两个在上、后两个在下，
    // 且第 0/3 个在世界 +y 一侧（相机视角的左）。
    check(
      plate.corners[0].z() > plate.corners[3].z() &&
        plate.corners[1].z() > plate.corners[2].z(),
      "corners 0/1 must be above corners 3/2");
    check(
      plate.corners[0].y() > plate.corners[1].y() &&
        plate.corners[3].y() > plate.corners[2].y(),
      "corners 0/3 must be on the camera-left side");

    // 板系：x 沿法线朝外，z 朝上。此处法线是 -x、上是 +z。
    const Eigen::Matrix3d rotation = plate.orientation.toRotationMatrix();
    check(
      (rotation.col(0) - Eigen::Vector3d{-1.0, 0.0, 0.0}).norm() < 1e-6,
      "plate x axis must be the outward normal");
    check(
      (rotation.col(2) - Eigen::Vector3d{0.0, 0.0, 1.0}).norm() < 1e-6,
      "plate z axis must point up");

    // 四个角点的均值必须等于发布的板中心，这是残差统计的基本前提。
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& corner : plate.corners) {
      mean += corner;
    }
    mean /= 4.0;
    check((mean - plate.center).norm() < 1e-6, "corners must average to centre");

    auto newest = source->latest();
    check(newest.has_value() && newest->frame_seq == 41, "latest() failed");

    // 环形槽位只有 kSlots 格，落后一整圈的帧必须取不到而不是取到错的。
    for (std::uint64_t seq = 42; seq <= 41 + kSlots; ++seq) {
      producer.publish(makeBatch(seq));
    }
    check(
      !source->forFrame(41).has_value(),
      "a frame overwritten by a full ring lap must not resolve");
    check(
      source->forFrame(41 + kSlots).has_value(),
      "the newest frame must still resolve after a full ring lap");

    // 写到一半的槽位必须被拒绝，而不是读出撕裂的数据。
    producer.beginTornWrite(50);
    check(
      !source->forFrame(50).has_value(),
      "a slot mid-write must not be readable");
    check(!source->takeLastError().empty(), "a torn read must report a reason");

    producer.setHeartbeat(1);
    check(!source->producerAlive(), "a stale heartbeat must read as dead");
  } catch (const std::exception& exception) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::cerr << "daedalus ground truth smoke test failed: " << exception.what()
              << '\n';
    return 1;
  }

  std::cout << "daedalus ground truth smoke test passed\n";
  return 0;
}
