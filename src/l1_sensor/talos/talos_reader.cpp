#include "l1_sensor/talos/talos_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <thread>
#include <utility>

namespace L1Sensor::talos {
namespace {
std::string joinPath(std::string directory, const char* name)
{
  while (!directory.empty() && directory.back() == '/') directory.pop_back();
  return directory + "/" + name;
}
} // namespace

TalosReader::TalosReader(std::string shm_dir) : shm_dir_(std::move(shm_dir)) {}
TalosReader::~TalosReader() { unmap(); }

void TalosReader::unmap()
{
  if (image_pool_ && image_pool_ != MAP_FAILED) munmap(image_pool_, kImagePoolSize);
  if (meta_ && meta_ != MAP_FAILED) munmap(meta_, sizeof(ShmMetaRegion));
  if (image_fd_ >= 0) close(image_fd_);
  if (meta_fd_ >= 0) close(meta_fd_);
  meta_ = image_pool_ = nullptr;
  region_ = nullptr;
  meta_fd_ = image_fd_ = -1;
  opened_ = false;
  camera_info_ = {};
  last_poses_ = {};
  last_chassis_ = {};
}

bool TalosReader::open()
{
  if (opened_) return true;
  meta_fd_ = ::open(joinPath(shm_dir_, kShmMetaName).c_str(), O_RDWR);
  image_fd_ = ::open(joinPath(shm_dir_, kShmImagePoolName).c_str(), O_RDWR);
  if (meta_fd_ < 0 || image_fd_ < 0) { unmap(); return false; }
  meta_ = mmap(nullptr, sizeof(ShmMetaRegion), PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd_, 0);
  image_pool_ = mmap(nullptr, kImagePoolSize, PROT_READ | PROT_WRITE, MAP_SHARED, image_fd_, 0);
  if (meta_ == MAP_FAILED || image_pool_ == MAP_FAILED) { unmap(); return false; }
  region_ = static_cast<ShmMetaRegion*>(meta_);
  if (region_->header.magic != kShmMagic || region_->header.version != kShmVersion ||
      region_->header.image_width != kImageWidth || region_->header.image_height != kImageHeight) {
    unmap();
    return false;
  }
  camera_info_ = region_->camera_info;
  steady_epoch_offset_ = std::chrono::steady_clock::now().time_since_epoch() -
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::nanoseconds{realtimeNowNs()});
  opened_ = true;
  return true;
}

bool TalosReader::reopen()
{
  unmap();
  return open();
}

bool TalosReader::consumeImage(TalosFrame& frame)
{
  auto& buffer = region_->image;
  std::uint8_t expected = __atomic_load_n(&buffer.state, __ATOMIC_ACQUIRE);
  if ((expected & kFlagNew) == 0) return false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    const std::uint8_t ready = expected & kIndexMask;
    const std::uint8_t desired = buffer.read_idx;
    if (__atomic_compare_exchange_n(&buffer.state, &expected, desired, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      buffer.read_idx = ready;
      const auto& meta = buffer.slots[ready];
      if (meta.buffer_id >= 3 || meta.format != 0 || meta.width != kImageWidth || meta.height != kImageHeight) {
        return false;
      }
      frame = {meta.seq, meta.timestamp_ns, meta.width, meta.height,
               static_cast<const std::uint8_t*>(image_pool_) +
                 static_cast<std::size_t>(meta.buffer_id) * kImageSize};
      return true;
    }
    if ((expected & kFlagNew) == 0) return false;
  }
  return false;
}

bool TalosReader::consumePose(PoseIndex index)
{
  auto& buffer = region_->poses[static_cast<std::size_t>(index)];
  std::uint8_t expected = __atomic_load_n(&buffer.state, __ATOMIC_ACQUIRE);
  if ((expected & kFlagNew) == 0) return false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    const std::uint8_t ready = expected & kIndexMask;
    const std::uint8_t desired = buffer.read_idx;
    if (__atomic_compare_exchange_n(&buffer.state, &expected, desired, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      buffer.read_idx = ready;
      last_poses_[static_cast<std::size_t>(index)] = buffer.slots[ready];
      return true;
    }
    if ((expected & kFlagNew) == 0) return false;
  }
  return false;
}

bool TalosReader::tryConsumeFrame(TalosFrame& frame)
{
  if (!consumeImage(frame)) return false;
  bool synchronized = true;
  for (std::uint8_t i = 0; i <= static_cast<std::uint8_t>(PoseIndex::Camera); ++i) {
    synchronized = consumePose(static_cast<PoseIndex>(i)) && synchronized;
  }
  last_chassis_ = region_->chassis_observation;
  if (!synchronized) return false;
  // 发布端承诺同一快照；序号不一致时宁可丢帧，不能把旧姿态配给新图像。
  for (std::uint8_t i = 0; i <= static_cast<std::uint8_t>(PoseIndex::Camera); ++i) {
    if (last_poses_[i].frame_seq != frame.seq) return false;
  }
  return true;
}

bool TalosReader::readFrame(TalosFrame& frame, std::chrono::milliseconds timeout)
{
  if (!opened_) return false;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (tryConsumeFrame(frame)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

std::optional<PoseMeta> TalosReader::pose(PoseIndex index) const noexcept
{
  const auto& value = last_poses_[static_cast<std::size_t>(index)];
  return value.timestamp_ns == 0 ? std::nullopt : std::optional<PoseMeta>{value};
}

GroundTruthBatch TalosReader::groundTruth() const noexcept
{
  return opened_ ? region_->ground_truth : GroundTruthBatch{};
}

RuntimeState TalosReader::runtimeState() const noexcept
{
  return opened_ ? region_->runtime_state : RuntimeState{};
}

void TalosReader::writeGimbalCmd(const GimbalCmd& command)
{
  if (!opened_) return;
  auto& buffer = region_->gimbal_cmd;
  GimbalCmd value = command;
  if (value.timestamp_ns == 0) value.timestamp_ns = realtimeNowNs();
  buffer.slots[buffer.write_idx] = value;
  const std::uint8_t old = __atomic_exchange_n(
    &buffer.state, buffer.write_idx | kFlagNew, __ATOMIC_ACQ_REL);
  buffer.write_idx = old & kIndexMask;
}

std::chrono::steady_clock::time_point TalosReader::toSteadyTime(std::uint64_t timestamp_ns) const noexcept
{
  const auto now = std::chrono::steady_clock::now();
  if (timestamp_ns == 0) return now;
  const auto mapped = std::chrono::steady_clock::time_point{
    steady_epoch_offset_ + std::chrono::nanoseconds{timestamp_ns}};
  return mapped > now + std::chrono::milliseconds{100} ? now : mapped;
}

std::uint64_t TalosReader::realtimeNowNs() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count());
}

} // namespace L1Sensor::talos
