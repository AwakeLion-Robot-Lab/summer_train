#include "l1_sensor/talos/talos_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

namespace L1Sensor::talos {
namespace {

std::string joinPath(std::string dir, const char* name)
{
  while (!dir.empty() && dir.back() == '/') {
    dir.pop_back();
  }
  return dir + "/" + name;
}

}  // namespace

TalosReader::TalosReader(std::string shm_dir)
  : shm_dir_(std::move(shm_dir))
{
}

TalosReader::~TalosReader()
{
  unmap();
}

void TalosReader::unmap()
{
  if (image_pool_ != nullptr && image_pool_ != MAP_FAILED) {
    munmap(image_pool_, kImagePoolSize);
  }
  if (meta_ != nullptr && meta_ != MAP_FAILED) {
    munmap(meta_, sizeof(ShmMetaRegion));
  }
  if (image_fd_ >= 0) {
    close(image_fd_);
  }
  if (meta_fd_ >= 0) {
    close(meta_fd_);
  }
  meta_ = nullptr;
  image_pool_ = nullptr;
  region_ = nullptr;
  meta_fd_ = -1;
  image_fd_ = -1;
  opened_ = false;
}

bool TalosReader::open()
{
  if (opened_) {
    return true;
  }
  meta_fd_ = ::open(joinPath(shm_dir_, kShmMetaName).c_str(), O_RDWR);
  image_fd_ = ::open(joinPath(shm_dir_, kShmImagePoolName).c_str(), O_RDWR);
  if (meta_fd_ < 0 || image_fd_ < 0) {
    unmap();
    return false;
  }
  meta_ = mmap(
    nullptr, sizeof(ShmMetaRegion), PROT_READ | PROT_WRITE,
    MAP_SHARED, meta_fd_, 0);
  image_pool_ = mmap(
    nullptr, kImagePoolSize, PROT_READ | PROT_WRITE,
    MAP_SHARED, image_fd_, 0);
  if (meta_ == MAP_FAILED || image_pool_ == MAP_FAILED) {
    unmap();
    return false;
  }
  region_ = static_cast<ShmMetaRegion*>(meta_);
  if (region_->header.magic != kShmMagic
      || region_->header.version != kShmVersion) {
    unmap();
    return false;
  }
  camera_info_ = region_->camera_info;

  // 标定 steady_clock 与 UNIX 纳秒时间戳的固定偏移，用于把 SHM 时间戳
  // 映射成 L3 需要的单调时钟。
  const auto steady_now = std::chrono::steady_clock::now();
  steady_epoch_offset_ =
    steady_now.time_since_epoch()
    - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::nanoseconds{realtimeNowNs()});
  opened_ = true;
  return true;
}

bool TalosReader::consumeImage(TalosFrame& frame)
{
  ImageTripleBuffer& buf = region_->image;
  std::uint8_t expected = __atomic_load_n(&buf.state, __ATOMIC_ACQUIRE);
  if ((expected & kFlagNew) == 0) {
    return false;
  }
  for (int attempt = 0; attempt < 2; ++attempt) {
    const std::uint8_t ready = expected & kIndexMask;
    const std::uint8_t desired = buf.read_idx;
    if (__atomic_compare_exchange_n(
          &buf.state, &expected, desired, false,
          __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      buf.read_idx = ready;
      const ImageMeta& meta = buf.slots[ready];
      frame.seq = meta.seq;
      frame.timestamp_ns = meta.timestamp_ns;
      frame.width = meta.width;
      frame.height = meta.height;
      frame.rgb =
        static_cast<const std::uint8_t*>(image_pool_)
        + static_cast<std::size_t>(meta.buffer_id) * kImageSize;
      return true;
    }
    if ((expected & kFlagNew) == 0) {
      return false;
    }
  }
  return false;
}

bool TalosReader::consumePose(PoseIndex index)
{
  PoseTripleBuffer& buf =
    region_->poses[static_cast<std::size_t>(index)];
  std::uint8_t expected = __atomic_load_n(&buf.state, __ATOMIC_ACQUIRE);
  if ((expected & kFlagNew) == 0) {
    return false;
  }
  for (int attempt = 0; attempt < 2; ++attempt) {
    const std::uint8_t ready = expected & kIndexMask;
    const std::uint8_t desired = buf.read_idx;
    if (__atomic_compare_exchange_n(
          &buf.state, &expected, desired, false,
          __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      buf.read_idx = ready;
      last_poses_[static_cast<std::size_t>(index)] = buf.slots[ready];
      return true;
    }
    if ((expected & kFlagNew) == 0) {
      return false;
    }
  }
  return false;
}

bool TalosReader::tryConsumeFrame(TalosFrame& frame)
{
  if (!consumeImage(frame)) {
    return false;
  }
  // 仿真器要求 image 与 Gimbal/Odom/Muzzle/Camera 四路 pose 全部被消费
  // 才发布下一帧；即使不需要位姿也必须走一遍消费。
  for (std::uint8_t i = 0;
       i <= static_cast<std::uint8_t>(PoseIndex::Camera); ++i) {
    consumePose(static_cast<PoseIndex>(i));
  }
  last_chassis_ = region_->chassis_observation;
  return true;
}

bool TalosReader::readFrame(
  TalosFrame& frame, std::chrono::milliseconds timeout)
{
  if (!opened_) {
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    if (tryConsumeFrame(frame)) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

std::optional<PoseMeta> TalosReader::pose(PoseIndex index) const noexcept
{
  const PoseMeta& pose =
    last_poses_[static_cast<std::size_t>(index)];
  if (pose.timestamp_ns == 0) {
    return std::nullopt;
  }
  return pose;
}

GroundTruthBatch TalosReader::groundTruth() const noexcept
{
  if (!opened_) {
    return {};
  }
  return region_->ground_truth;
}

void TalosReader::writeGimbalCmd(const GimbalCmd& cmd)
{
  if (!opened_) {
    return;
  }
  GimbalTripleBuffer& buf = region_->gimbal_cmd;
  GimbalCmd value = cmd;
  if (value.timestamp_ns == 0) {
    value.timestamp_ns = realtimeNowNs();
  }
  buf.slots[buf.write_idx] = value;
  const std::uint8_t old =
    __atomic_exchange_n(&buf.state, buf.write_idx | kFlagNew, __ATOMIC_ACQ_REL);
  buf.write_idx = old & kIndexMask;
}

std::chrono::steady_clock::time_point TalosReader::toSteadyTime(
  std::uint64_t timestamp_ns) const noexcept
{
  const auto now = std::chrono::steady_clock::now();
  if (timestamp_ns == 0) {
    return now;
  }
  const auto mapped = std::chrono::steady_clock::time_point{
    steady_epoch_offset_ + std::chrono::nanoseconds{timestamp_ns}};
  if (mapped > now + std::chrono::milliseconds{100}) {
    return now;
  }
  return mapped;
}

std::uint64_t TalosReader::realtimeNowNs() noexcept
{
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

}  // namespace L1Sensor::talos
