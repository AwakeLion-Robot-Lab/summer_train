#include "l1_sensor/daedalus_ground_truth.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <numbers>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace L1Sensor {
namespace {

// 与 crates/talos-ipc/src/plate_truth.rs 一一对应。改那边就必须改这里，
// 静态断言是唯一的护栏。
constexpr std::uint32_t kMagic = 0x54505401;
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kMaxTargets = 16;
constexpr std::size_t kMaxPlates = 64;
constexpr std::size_t kSlots = 8;
constexpr std::uint32_t kUnlinked = 0xFFFFFFFFU;
constexpr std::uint8_t kUnlinkedIndex = 0xFFU;

// seqlock 重试上限。生产者按帧率写（最快 ~200 Hz），一次 9KB 的 memcpy 是
// 微秒量级，正常情况下第一次就成功；给足重试只是为了不在调度抖动时误报。
constexpr int kMaxSeqlockRetries = 64;

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
static_assert(sizeof(PlateTruth) == 128);
static_assert(offsetof(PlateTruth, center) == 8);
static_assert(offsetof(PlateTruth, quaternion) == 20);
static_assert(offsetof(PlateTruth, corners) == 36);
static_assert(offsetof(PlateTruth, yaw) == 84);
static_assert(offsetof(PlateTruth, radius) == 92);

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
static_assert(sizeof(GroundTruthTarget) == 64);

struct alignas(64) PlateTruthBatch {
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t target_count;
  std::uint32_t plate_count;
  GroundTruthTarget targets[kMaxTargets];
  PlateTruth plates[kMaxPlates];
};
static_assert(sizeof(PlateTruthBatch) == 9280);
static_assert(offsetof(PlateTruthBatch, targets) == 32);
static_assert(offsetof(PlateTruthBatch, plates) == 1088);

struct alignas(64) PlateTruthSlot {
  alignas(64) std::atomic<std::uint64_t> seq;
  std::uint8_t padding[56];
  PlateTruthBatch batch;
};
static_assert(sizeof(PlateTruthSlot) == 9344);
static_assert(offsetof(PlateTruthSlot, batch) == 64);

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
static_assert(sizeof(PlateTruthHeader) == 64);

struct PlateTruthRegion {
  PlateTruthHeader header;
  PlateTruthSlot slots[kSlots];
};
static_assert(sizeof(PlateTruthRegion) == 74816);
static_assert(offsetof(PlateTruthRegion, slots) == 64);

[[nodiscard]] std::uint64_t systemNowNs() noexcept
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

[[nodiscard]] std::uint64_t loadSharedU64(const std::uint64_t& value) noexcept
{
  return __atomic_load_n(&value, __ATOMIC_ACQUIRE);
}

class MappedFile {
public:
  [[nodiscard]] static std::unique_ptr<MappedFile> open(
    const std::string& path,
    std::size_t required_size,
    std::string& error)
  {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      error = "cannot open " + path + ": " + std::strerror(errno);
      return nullptr;
    }

    struct stat status {};
    if (::fstat(fd, &status) != 0) {
      error = "cannot stat " + path + ": " + std::strerror(errno);
      ::close(fd);
      return nullptr;
    }
    if (status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) < required_size) {
      error = path + " is smaller than the plate-truth protocol requires";
      ::close(fd);
      return nullptr;
    }

    void* data = ::mmap(nullptr, required_size, PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
      error = "cannot mmap " + path + ": " + std::strerror(errno);
      ::close(fd);
      return nullptr;
    }
    return std::unique_ptr<MappedFile>(new MappedFile(fd, data, required_size));
  }

  ~MappedFile()
  {
    if (data_ != nullptr && data_ != MAP_FAILED) {
      ::munmap(data_, size_);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  [[nodiscard]] const void* data() const noexcept { return data_; }

private:
  MappedFile(int fd, void* data, std::size_t size)
    : fd_(fd), data_(data), size_(size)
  {
  }

  int fd_{-1};
  void* data_{nullptr};
  std::size_t size_{0};
};

// seqlock 读取：生产者写之前把 seq 变奇数，写完再变偶数。前后两次读到同一个
// 偶数才说明这 9KB 没有被写穿。
[[nodiscard]] bool readSlot(const PlateTruthSlot& slot, PlateTruthBatch& out) noexcept
{
  for (int attempt = 0; attempt < kMaxSeqlockRetries; ++attempt) {
    const std::uint64_t before = slot.seq.load(std::memory_order_acquire);
    if ((before & 1U) != 0U) {
      continue;
    }
    std::memcpy(&out, &slot.batch, sizeof(PlateTruthBatch));
    std::atomic_thread_fence(std::memory_order_acquire);
    if (slot.seq.load(std::memory_order_acquire) == before) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] int decodeTeam(std::uint8_t team) noexcept
{
  return team <= 1 ? static_cast<int>(team) : -1;
}

[[nodiscard]] bool finiteArray(const float* values, std::size_t count) noexcept
{
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<DaedalusTargetTruth> convertTarget(
  const GroundTruthTarget& raw)
{
  if (!finiteArray(raw.position, 3) || !std::isfinite(raw.yaw) ||
      !std::isfinite(raw.vyaw)) {
    return std::nullopt;
  }
  DaedalusTargetTruth target;
  target.team = decodeTeam(raw.team);
  target.armor_label = static_cast<int>(raw.armor_label);
  target.position = Eigen::Vector3d{
    raw.position[0], raw.position[1], raw.position[2]};
  target.yaw = raw.yaw;
  target.v_yaw = raw.vyaw;
  return target;
}

[[nodiscard]] std::optional<DaedalusPlateTruth> convertPlate(
  const PlateTruth& raw,
  std::uint32_t target_count)
{
  if (!finiteArray(raw.center, 3) || !finiteArray(raw.quaternion, 4) ||
      !finiteArray(&raw.corners[0][0], 12) || !std::isfinite(raw.yaw) ||
      !std::isfinite(raw.pitch) || !std::isfinite(raw.radius)) {
    return std::nullopt;
  }

  Eigen::Quaterniond orientation{
    raw.quaternion[0], raw.quaternion[1], raw.quaternion[2], raw.quaternion[3]};
  if (orientation.squaredNorm() <= 1e-12) {
    return std::nullopt;
  }
  orientation.normalize();

  DaedalusPlateTruth plate;
  // 越界的 target_index 当作未关联，而不是相信它去索引 targets。
  plate.target_index =
    (raw.target_index != kUnlinked && raw.target_index < target_count)
      ? static_cast<int>(raw.target_index)
      : -1;
  plate.team = decodeTeam(raw.team);
  plate.armor_label = static_cast<int>(raw.armor_label);
  plate.armor_size = raw.armor_type == 1 ? DaedalusArmorSize::Large
                                         : DaedalusArmorSize::Small;
  plate.plate_index = raw.plate_index == kUnlinkedIndex
                        ? -1
                        : static_cast<int>(raw.plate_index);
  plate.center = Eigen::Vector3d{raw.center[0], raw.center[1], raw.center[2]};
  plate.orientation = orientation;
  for (std::size_t index = 0; index < plate.corners.size(); ++index) {
    plate.corners[index] = Eigen::Vector3d{
      raw.corners[index][0], raw.corners[index][1], raw.corners[index][2]};
  }
  plate.yaw = raw.yaw;
  plate.pitch = raw.pitch;
  plate.radius = raw.radius;
  return plate;
}

}  // namespace

double toNewvisionArmorYaw(double outward_yaw) noexcept
{
  if (!std::isfinite(outward_yaw)) {
    return outward_yaw;
  }
  double yaw = outward_yaw + std::numbers::pi;
  yaw = std::fmod(yaw + std::numbers::pi, 2.0 * std::numbers::pi);
  if (yaw <= 0.0) {
    yaw += 2.0 * std::numbers::pi;
  }
  return yaw - std::numbers::pi;
}

struct DaedalusGroundTruthSource::Impl {
  Impl(DaedalusGroundTruthOptions source_options,
       std::unique_ptr<MappedFile> mapping)
    : options(std::move(source_options)),
      region_mapping(std::move(mapping)),
      region(static_cast<const PlateTruthRegion*>(region_mapping->data()))
  {
  }

  void setError(std::string message) const { last_error = std::move(message); }

  DaedalusGroundTruthOptions options;
  std::unique_ptr<MappedFile> region_mapping;
  const PlateTruthRegion* region{nullptr};
  mutable std::string last_error;
};

DaedalusGroundTruthSource::DaedalusGroundTruthSource(std::unique_ptr<Impl> impl)
  : impl_(std::move(impl))
{
}

DaedalusGroundTruthSource::~DaedalusGroundTruthSource() = default;

std::unique_ptr<DaedalusGroundTruthSource> DaedalusGroundTruthSource::connect(
  const DaedalusGroundTruthOptions& options,
  std::string* error)
{
  std::string detail;
  auto mapping = MappedFile::open(
    options.path, sizeof(PlateTruthRegion), detail);
  if (!mapping) {
    if (error != nullptr) {
      *error = std::move(detail);
    }
    return nullptr;
  }

  const auto* region = static_cast<const PlateTruthRegion*>(mapping->data());
  if (region->header.magic != kMagic) {
    detail = "Daedalus plate-truth magic mismatch";
  } else if (region->header.version != kVersion) {
    detail = "Daedalus plate-truth version mismatch: expected " +
      std::to_string(kVersion) + ", got " +
      std::to_string(region->header.version);
  } else if (region->header.max_targets != kMaxTargets ||
             region->header.max_plates != kMaxPlates ||
             region->header.slot_count != kSlots) {
    detail = "Daedalus plate-truth capacities do not match this build";
  }
  if (!detail.empty()) {
    if (error != nullptr) {
      *error = std::move(detail);
    }
    return nullptr;
  }

  auto impl = std::make_unique<Impl>(options, std::move(mapping));
  return std::unique_ptr<DaedalusGroundTruthSource>(
    new DaedalusGroundTruthSource(std::move(impl)));
}

bool DaedalusGroundTruthSource::producerAlive() const noexcept
{
  const std::uint64_t heartbeat_ns =
    loadSharedU64(impl_->region->header.heartbeat_ns);
  if (heartbeat_ns == 0) {
    return false;
  }
  const auto timeout_count =
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      impl_->options.producer_timeout).count();
  const std::uint64_t now_ns = systemNowNs();
  if (timeout_count <= 0 || now_ns < heartbeat_ns) {
    return false;
  }
  return now_ns - heartbeat_ns <= static_cast<std::uint64_t>(timeout_count);
}

std::size_t DaedalusGroundTruthSource::slotCount() const noexcept
{
  return kSlots;
}

std::optional<DaedalusGroundTruth> DaedalusGroundTruthSource::forFrame(
  std::uint64_t frame_seq) const
{
  const PlateTruthSlot& slot = impl_->region->slots[frame_seq % kSlots];

  PlateTruthBatch batch{};
  if (!readSlot(slot, batch)) {
    impl_->setError("plate-truth slot kept changing while being read");
    return std::nullopt;
  }
  if (batch.frame_seq != frame_seq) {
    // 槽位已经被后来的帧覆盖，或者这一帧还没发布。宁可没有真值，
    // 也不能把相邻帧的真值当成这一帧的。
    return std::nullopt;
  }
  if (batch.target_count > kMaxTargets || batch.plate_count > kMaxPlates) {
    impl_->setError("plate-truth snapshot declared an out-of-range count");
    return std::nullopt;
  }

  DaedalusGroundTruth result;
  result.frame_seq = batch.frame_seq;
  result.timestamp_ns = batch.timestamp_ns;

  result.targets.reserve(batch.target_count);
  for (std::uint32_t index = 0; index < batch.target_count; ++index) {
    auto target = convertTarget(batch.targets[index]);
    if (!target) {
      impl_->setError("plate-truth snapshot contained a non-finite target");
      return std::nullopt;
    }
    result.targets.push_back(*target);
  }

  result.plates.reserve(batch.plate_count);
  for (std::uint32_t index = 0; index < batch.plate_count; ++index) {
    auto plate = convertPlate(batch.plates[index], batch.target_count);
    if (!plate) {
      impl_->setError("plate-truth snapshot contained an invalid plate");
      return std::nullopt;
    }
    result.plates.push_back(*plate);
  }
  return result;
}

std::optional<DaedalusGroundTruth> DaedalusGroundTruthSource::latest() const
{
  const std::uint64_t frame_seq =
    impl_->region->header.latest_frame_seq.load(std::memory_order_acquire);
  return forFrame(frame_seq);
}

std::string DaedalusGroundTruthSource::takeLastError()
{
  std::string result = std::move(impl_->last_error);
  impl_->last_error.clear();
  return result;
}

}  // namespace L1Sensor
