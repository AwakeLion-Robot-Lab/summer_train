#pragma once

#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace runtime {

// One producer, one consumer. Unlike LatestBuffer, publishing waits for consumption
// BEFORE reading/processing another frame. At most one frame runs ahead of the
// consumer. Produce returns nullopt at end of input and must have bounded waits.
template<class T>
class SingleFramePipeline {
public:
  enum class ReadStatus { Value, Timeout, Closed };

  template<class Produce>
  explicit SingleFramePipeline(Produce produce)
    : worker_([this, produce = std::move(produce)](std::stop_token stop) mutable {
        try {
          while (!stop.stop_requested()) {
            auto next = produce();
            if (!next) break;
            std::unique_lock lock(mutex_);
            if (stop.stop_requested()) break;
            slot_ = std::move(*next);
            cv_.notify_all();
            if (!cv_.wait(lock, stop, [this] { return !slot_; })) break;
          }
        } catch (...) {
          std::lock_guard lock(mutex_);
          error_ = std::current_exception();
        }
        {
          std::lock_guard lock(mutex_);
          closed_ = true;
        }
        cv_.notify_all();
      }) {}

  ~SingleFramePipeline() { stop(); }
  SingleFramePipeline(const SingleFramePipeline&) = delete;
  SingleFramePipeline& operator=(const SingleFramePipeline&) = delete;

  template<class Rep, class Period>
  ReadStatus readFor(T& out, std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] { return slot_ || closed_; }))
      return ReadStatus::Timeout;
    if (error_) std::rethrow_exception(error_);
    if (!slot_) return ReadStatus::Closed;
    out = std::move(*slot_);
    slot_.reset();
    lock.unlock();
    cv_.notify_all();
    return ReadStatus::Value;
  }

  // Consumer/lifecycle thread only. Joining before destroying detector/tracker,
  // camera and serial prevents borrowed production dependencies outliving run().
  void stop() {
    worker_.request_stop();
    if (worker_.joinable()) worker_.join();
    std::lock_guard lock(mutex_);
    slot_.reset();
  }

private:
  std::mutex mutex_;
  std::condition_variable_any cv_;
  std::optional<T> slot_;
  std::exception_ptr error_;
  bool closed_{false};
  // Last member: all synchronization state exists before the worker starts.
  std::jthread worker_;
};

} // namespace runtime
