#include "serial/serial.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint8_t kSof = 0xA0;
constexpr std::uint16_t kPingCmd = 0x7F01;
constexpr std::uint16_t kPongCmd = 0x7F02;
constexpr std::size_t kHeaderSize = 7;
constexpr std::size_t kTokenSize = 4;
constexpr std::size_t kFrameSize = kHeaderSize + kTokenSize + 2;
constexpr std::size_t kMaxPayloadSize = 256;

struct Options {
  std::string port = "/dev/ttyACM0";
  unsigned baud = 115200;
  unsigned count = 1000;
  unsigned interval_ms = 20;
  unsigned timeout_ms = 100;
  unsigned hold_ms = 20;
  unsigned max_consecutive_timeouts = 10;
};

template <typename T>
bool parsePositive(std::string_view input, T& output)
{
  T value{};
  const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc{} || end != input.data() + input.size() || value == 0) return false;
  output = value;
  return true;
}

bool parseNonNegative(std::string_view input, unsigned& output)
{
  const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), output);
  return error == std::errc{} && end == input.data() + input.size();
}

std::uint8_t crc8(std::span<const std::uint8_t> bytes)
{
  std::uint8_t crc = 0xFF;
  for (const auto byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1U) != 0 ? static_cast<std::uint8_t>((crc >> 1) ^ 0x8C)
                             : static_cast<std::uint8_t>(crc >> 1);
    }
  }
  return crc;
}

std::uint16_t crc16(std::span<const std::uint8_t> bytes)
{
  std::uint16_t crc = 0xFFFF;
  for (const auto byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1U) != 0 ? static_cast<std::uint16_t>((crc >> 1) ^ 0x8408)
                             : static_cast<std::uint16_t>(crc >> 1);
    }
  }
  return crc;
}

std::array<std::uint8_t, kFrameSize> makePing(std::uint8_t seq, std::uint32_t token)
{
  std::array<std::uint8_t, kFrameSize> frame{};
  frame[0] = kSof;
  frame[1] = static_cast<std::uint8_t>(kTokenSize);
  frame[2] = 0;
  frame[3] = seq;
  frame[4] = crc8(std::span<const std::uint8_t>{frame.data(), 4});
  frame[5] = static_cast<std::uint8_t>(kPingCmd);
  frame[6] = static_cast<std::uint8_t>(kPingCmd >> 8);
  for (int i = 0; i < 4; ++i) frame[7 + i] = static_cast<std::uint8_t>(token >> (8 * i));
  const auto checksum = crc16(std::span<const std::uint8_t>{frame.data(), kFrameSize - 2});
  frame[kFrameSize - 2] = static_cast<std::uint8_t>(checksum);
  frame[kFrameSize - 1] = static_cast<std::uint8_t>(checksum >> 8);
  return frame;
}

class PongParser {
public:
  std::optional<std::uint32_t> feed(std::uint8_t byte)
  {
    bytes_.push_back(byte);
    while (!bytes_.empty()) {
      if (bytes_.front() != kSof) {
        bytes_.erase(bytes_.begin());
        continue;
      }
      if (bytes_.size() < kHeaderSize) return std::nullopt;
      if (crc8(std::span<const std::uint8_t>{bytes_.data(), 4}) != bytes_[4]) {
        ++bad_crc_;
        bytes_.erase(bytes_.begin());
        continue;
      }
      const auto payload_size = static_cast<std::size_t>(bytes_[1] | (bytes_[2] << 8));
      if (payload_size > kMaxPayloadSize) {
        bytes_.erase(bytes_.begin());
        continue;
      }
      const auto frame_size = kHeaderSize + payload_size + 2;
      if (bytes_.size() < frame_size) return std::nullopt;
      const auto expected_crc = static_cast<std::uint16_t>(
        bytes_[frame_size - 2] | (bytes_[frame_size - 1] << 8));
      if (crc16(std::span<const std::uint8_t>{bytes_.data(), frame_size - 2}) != expected_crc) {
        ++bad_crc_;
        bytes_.erase(bytes_.begin());
        continue;
      }
      const auto cmd = static_cast<std::uint16_t>(bytes_[5] | (bytes_[6] << 8));
      std::optional<std::uint32_t> token;
      if (cmd == kPongCmd && payload_size == kTokenSize) {
        token = static_cast<std::uint32_t>(bytes_[7]) |
                (static_cast<std::uint32_t>(bytes_[8]) << 8) |
                (static_cast<std::uint32_t>(bytes_[9]) << 16) |
                (static_cast<std::uint32_t>(bytes_[10]) << 24);
      }
      bytes_.erase(bytes_.begin(), bytes_.begin() + frame_size);
      if (token) return token;
    }
    return std::nullopt;
  }

  std::uint64_t badCrcCount() const { return bad_crc_; }

private:
  std::vector<std::uint8_t> bytes_;
  std::uint64_t bad_crc_ = 0;
};

double percentile(const std::vector<double>& sorted, double fraction)
{
  const double index = fraction * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(index);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

std::optional<Clock::time_point> waitForPong(serial::Serial& port, PongParser& parser,
                                             std::uint32_t token, Clock::time_point deadline,
                                             unsigned& unmatched)
{
  while (Clock::now() < deadline) {
    std::uint8_t byte = 0;
    if (port.read(&byte, 1) != 1) continue;
    const auto received_at = Clock::now();
    const auto response = parser.feed(byte);
    if (!response) continue;
    if (*response == token) return received_at;
    ++unmatched;
  }
  return std::nullopt;
}

int verifyFrameHandling(serial::Serial& port, PongParser& parser,
                        std::uint32_t first_token, const Options& options)
{
  unsigned unmatched = 0;
  const auto timeout = std::chrono::milliseconds(options.timeout_ms);
  const auto hold = std::chrono::milliseconds(options.hold_ms);
  std::cout << "verify Ping handling: " << options.port << " @ " << options.baud
            << ", timeout=" << options.timeout_ms << " ms, hold=" << options.hold_ms << " ms\n";

  const auto baseline = makePing(0, first_token);
  if (port.write(baseline.data(), baseline.size()) != baseline.size()) {
    std::cerr << "Serial write failed for baseline Ping\n";
    return 3;
  }
  if (!waitForPong(port, parser, first_token, Clock::now() + timeout, unmatched)) {
    std::cerr << "INCONCLUSIVE: a complete valid Ping did not receive Pong.\n";
    return 4;
  }
  std::cout << "PASS complete valid Ping returned Pong\n";

  const auto partial_token = first_token + 1;
  const auto partial = makePing(1, partial_token);
  if (port.write(partial.data(), partial.size() - 1) != partial.size() - 1) {
    std::cerr << "Serial write failed for partial Ping\n";
    return 3;
  }
  if (waitForPong(port, parser, partial_token, Clock::now() + hold, unmatched)) {
    std::cerr << "FAIL: Pong arrived before the final CRC16 byte was sent.\n";
    return 5;
  }
  std::cout << "PASS no Pong while the final CRC16 byte was withheld\n";

  if (port.write(partial.data() + partial.size() - 1, 1) != 1) {
    std::cerr << "Serial write failed for final CRC16 byte\n";
    return 3;
  }
  if (!waitForPong(port, parser, partial_token, Clock::now() + timeout, unmatched)) {
    std::cerr << "INCONCLUSIVE: no Pong after completing the split Ping; "
                 "the firmware may discard frames after an inter-byte timeout.\n";
    return 6;
  }
  std::cout << "PASS Pong arrived after the final byte\n";

  const auto corrupt_token = first_token + 2;
  auto corrupt = makePing(2, corrupt_token);
  corrupt.back() ^= 0x01;
  if (port.write(corrupt.data(), corrupt.size()) != corrupt.size()) {
    std::cerr << "Serial write failed for corrupt Ping\n";
    return 3;
  }
  if (waitForPong(port, parser, corrupt_token, Clock::now() + hold, unmatched)) {
    std::cerr << "FAIL: Pong arrived for a Ping with invalid CRC16.\n";
    return 5;
  }
  std::cout << "PASS no Pong for invalid CRC16\n";

  const auto final_token = first_token + 3;
  const auto final_ping = makePing(3, final_token);
  if (port.write(final_ping.data(), final_ping.size()) != final_ping.size()) {
    std::cerr << "Serial write failed for final valid Ping\n";
    return 3;
  }
  if (!waitForPong(port, parser, final_token, Clock::now() + timeout, unmatched)) {
    std::cerr << "INCONCLUSIVE: no Pong after the corrupt frame; "
                 "the firmware parser may need to resynchronize.\n";
    return 6;
  }
  std::cout << "PASS valid Ping still works after invalid CRC16\n"
            << "Verification passed (unmatched_pong=" << unmatched
            << ", bad_rx_crc=" << parser.badCrcCount() << ")\n";
  return 0;
}

void printUsage(const char* program)
{
  std::cerr << "Usage: " << program
            << " [port] [baud=115200] [count=1000] [interval_ms=20]"
               " [timeout_ms=100] [max_consecutive_timeouts=10]\n"
            << "       " << program
            << " verify [port] [baud=115200] [timeout_ms=100] [hold_ms=20]\n";
}

}  // namespace

int main(int argc, char* argv[])
{
  Options options;
  // xmake run passes a literal "--" through to the target; tolerate it for
  // commands copied from tools that use "--" as an argument separator.
  int first_arg = argc > 1 && std::string_view(argv[1]) == "--" ? 2 : 1;
  const bool verify_mode = argc > first_arg && std::string_view(argv[first_arg]) == "verify";
  if (verify_mode) ++first_arg;
  const int positional_count = argc - first_arg;
  const bool invalid_arguments = verify_mode
    ? (positional_count > 4 ||
       (positional_count > 1 && !parsePositive(argv[first_arg + 1], options.baud)) ||
       (positional_count > 2 && !parsePositive(argv[first_arg + 2], options.timeout_ms)) ||
       (positional_count > 3 && !parsePositive(argv[first_arg + 3], options.hold_ms)))
    : (positional_count > 6 ||
       (positional_count > 1 && !parsePositive(argv[first_arg + 1], options.baud)) ||
       (positional_count > 2 && !parsePositive(argv[first_arg + 2], options.count)) ||
       (positional_count > 3 && !parsePositive(argv[first_arg + 3], options.interval_ms)) ||
       (positional_count > 4 && !parsePositive(argv[first_arg + 4], options.timeout_ms)) ||
       (positional_count > 5 &&
        !parseNonNegative(argv[first_arg + 5], options.max_consecutive_timeouts)));
  if (invalid_arguments) {
    printUsage(argv[0]);
    return 2;
  }
  if (positional_count > 0) options.port = argv[first_arg];

  try {
    // Read exactly one byte per call so a large read buffer does not add a
    // timeout to the timestamp of an otherwise complete Pong frame.
    serial::Timeout timeout(1, 2, 0, options.timeout_ms, 0);
    serial::Serial port(options.port, options.baud, timeout);
    port.flushInput();

    std::random_device random;
    const std::uint32_t first_token = (static_cast<std::uint32_t>(random()) << 16) ^ random();
    PongParser parser;
    if (verify_mode) return verifyFrameHandling(port, parser, first_token, options);

    std::vector<double> samples;
    samples.reserve(options.count);
    unsigned timed_out = 0;
    unsigned unmatched = 0;
    unsigned attempted = 0;
    unsigned consecutive_timeouts = 0;
    bool stopped_early = false;

    std::cout << "serial RTT: " << options.port << " @ " << options.baud
              << ", probes=" << options.count << ", interval=" << options.interval_ms
              << " ms, timeout=" << options.timeout_ms
              << " ms, max_consecutive_timeouts=" << options.max_consecutive_timeouts << '\n';

    for (unsigned i = 0; i < options.count; ++i) {
      const auto token = first_token + i;
      const auto frame = makePing(static_cast<std::uint8_t>(i), token);
      const auto sent_at = Clock::now();
      if (port.write(frame.data(), frame.size()) != frame.size()) {
        std::cerr << "Serial write failed at probe " << i << '\n';
        return 3;
      }

      const auto deadline = sent_at + std::chrono::milliseconds(options.timeout_ms);
      if (const auto received_at = waitForPong(port, parser, token, deadline, unmatched)) {
        samples.push_back(std::chrono::duration<double, std::milli>(*received_at - sent_at).count());
        consecutive_timeouts = 0;
      } else {
        ++timed_out;
        ++consecutive_timeouts;
        if (timed_out <= 5) {
          std::cerr << "Timeout at probe " << i
                    << " (seq=" << static_cast<unsigned>(static_cast<std::uint8_t>(i)) << ")\n";
        }
      }
      ++attempted;
      if (attempted % 100 == 0) {
        std::cout << "progress: attempted=" << attempted << '/' << options.count
                  << " received=" << samples.size() << " timeout=" << timed_out
                  << " consecutive_timeout=" << consecutive_timeouts << std::endl;
      }
      if (options.max_consecutive_timeouts != 0 &&
          consecutive_timeouts >= options.max_consecutive_timeouts) {
        std::cerr << "Stopped after " << consecutive_timeouts
                  << " consecutive timeouts at probe " << i << ".\n";
        stopped_early = true;
        break;
      }
      std::this_thread::sleep_until(sent_at + std::chrono::milliseconds(options.interval_ms));
    }

    std::sort(samples.begin(), samples.end());
    std::cout << "attempted=" << attempted << '/' << options.count
              << " received=" << samples.size() << " timeout=" << timed_out
              << " unmatched_pong=" << unmatched << " bad_crc=" << parser.badCrcCount() << '\n';
    if (samples.empty()) {
      std::cerr << "No valid Pong; check firmware Ping/Pong support and serial settings.\n";
      return 4;
    }
    double sum = 0;
    for (const double value : samples) sum += value;
    std::cout << std::fixed << std::setprecision(3)
              << "RTT ms: min=" << samples.front()
              << " mean=" << sum / samples.size()
              << " p50=" << percentile(samples, 0.50)
              << " p95=" << percentile(samples, 0.95)
              << " p99=" << percentile(samples, 0.99)
              << " max=" << samples.back() << '\n';
    return stopped_early ? 4 : 0;
  } catch (const std::exception& error) {
    std::cerr << "Serial RTT failed: " << error.what() << '\n';
    return 3;
  }
}
