#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

namespace tools {

template <typename T>
class LatestBuffer {
public:
  enum class ReadStatus {
    Value,
    Timeout,
    Closed,
  };

  bool write(T value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_) {
      return false;
    }
    if (value_.has_value()) {
      ++dropped_count_;
    }
    value_ = std::move(value);
    lock.unlock();
    cv_.notify_one();
    return true;
  }

  bool read(T& out)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      return value_.has_value() || closed_;
    });
    if (!value_.has_value() && closed_) {
      return false;
    }
    out = std::move(*value_);
    value_.reset();
    return true;
  }

  template <typename Rep, typename Period>
  ReadStatus readFor(
    T& out, const std::chrono::duration<Rep, Period>& timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] {
          return value_.has_value() || closed_;
        })) {
      return ReadStatus::Timeout;
    }

    if (!value_.has_value()) {
      return ReadStatus::Closed;
    }

    out = std::move(*value_);
    value_.reset();
    return ReadStatus::Value;
  }

  void close()
  {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
  }

  bool closed() const
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return closed_;
  }

  uint64_t droppedCount() const
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return dropped_count_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::optional<T> value_;
  bool closed_ = false;
  uint64_t dropped_count_ = 0;
};

}  // namespace tools
