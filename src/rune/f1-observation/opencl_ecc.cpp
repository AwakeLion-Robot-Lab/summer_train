#include "opencl_ecc.hpp"

#include <opencv2/core/ocl.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace Rune::detail {
namespace {
struct Task {
  std::int32_t x, y, width, height, offset, parameters, iterations;
  float warp[6];
};
static_assert(sizeof(Task) == 13 * 4);
constexpr std::size_t work_group_size = 64;
using cv::ocl::KernelArg;

// ECC 的目标函数与 OpenCV 一致。每个工作组独立迭代一个局部区域。
constexpr char source[] = R"CLC(
#define LANES 64
#define STATS 45
typedef struct {
  int x, y, width, height, offset, parameters, iterations;
  float warp[6];
} Task;

int reflect(int x, int n) {
  return x < 0 ? -x : x >= n ? 2 * n - 2 - x : x;
}

__kernel void blur_regions(__global const uchar* before, int before_step, int before_offset,
    __global const uchar* after, int after_step, int after_offset,
    __global const Task* tasks, __global float2* blurred) {
  const Task t = tasks[get_global_id(1)];
  const int i = get_global_id(0);
  if (i >= t.width * t.height) return;
  const int x = i % t.width, y = i / t.width;
  const int weights[5] = {1, 4, 6, 4, 1};
  float2 sum = (float2)(0);
  for (int dy = -2; dy <= 2; ++dy) {
    const int row = t.y + reflect(y + dy, t.height);
    for (int dx = -2; dx <= 2; ++dx) {
      const int col = t.x + reflect(x + dx, t.width);
      const float2 pixels = (float2)(before[before_offset + row * before_step + col],
                                    after[after_offset + row * after_step + col]);
      sum += pixels * (weights[dy + 2] * weights[dx + 2]);
    }
  }
  blurred[t.offset + i] = sum * (1.0f / (255.0f * 256.0f));
}

__kernel void prepare_gradients(__global const Task* tasks,
    __global const float2* blurred, __global float4* pixels) {
  const Task t = tasks[get_global_id(1)];
  const int i = get_global_id(0);
  if (i >= t.width * t.height) return;
  const int x = i % t.width, y = i / t.width;
  __global const float2* img = blurred + t.offset;
  const float gx = .5f * (img[y * t.width + reflect(x + 1, t.width)].y -
                           img[y * t.width + reflect(x - 1, t.width)].y);
  const float gy = .5f * (img[reflect(y + 1, t.height) * t.width + x].y -
                           img[reflect(y - 1, t.height) * t.width + x].y);
  pixels[t.offset + i] = (float4)(img[i], gx, gy);
}

float3 sample(__global const float4* pixels, int w, int h, int x, int y) {
  return x < 0 || x >= w || y < 0 || y >= h ? (float3)(0) : pixels[y * w + x].yzw;
}

void solve(__private const float* lower, __private float* vector, int n) {
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < j; ++k) vector[j] -= lower[6 * j + k] * vector[k];
    vector[j] /= lower[6 * j + j];
  }
  for (int j = n - 1; j >= 0; --j) {
    for (int k = j + 1; k < n; ++k) vector[j] -= lower[6 * k + j] * vector[k];
    vector[j] /= lower[6 * j + j];
  }
}

float update(__local const float* totals, int n, int w, int h, __local float* warp) {
  const float count = totals[0];
  if (count < 16) return -INFINITY;
  const float mean_i = totals[1] / count, mean_t = totals[2] / count;
  const float norm_i = totals[3] - mean_i * totals[1];
  const float norm_t = totals[4] - mean_t * totals[2];
  const float correlation = totals[5] - mean_i * totals[2];
  if (norm_i < 1e-6f || norm_t < 1e-6f) return -INFINITY;
  float lower[36];
  for (int j = 0, k = 24; j < n; ++j) {
    for (int i = 0; i <= j; ++i, ++k) {
      float value = totals[k];
      for (int p = 0; p < i; ++p) value -= lower[6 * j + p] * lower[6 * i + p];
      if (i == j && !(value > 1e-10f)) return -INFINITY;
      lower[6 * j + i] = i == j ? sqrt(value) : value / lower[6 * i + i];
    }
  }
  float projection_i[6], projection_t[6], direction[6];
  for (int j = 0; j < n; ++j) {
    projection_i[j] = totals[12 + j] - mean_i * totals[6 + j];
    projection_t[j] = totals[18 + j] - mean_t * totals[6 + j];
    direction[j] = projection_i[j];
  }
  solve(lower, direction, n);
  float numerator = norm_i, denominator = correlation;
  for (int j = 0; j < n; ++j) {
    numerator -= projection_i[j] * direction[j];
    denominator -= projection_t[j] * direction[j];
  }
  if (!(denominator > 0)) return -INFINITY;
  const float gain = numerator / denominator;
  for (int j = 0; j < n; ++j) direction[j] = gain * projection_t[j] - projection_i[j];
  solve(lower, direction, n);
  for (int j = 0; j < n; ++j) if (!isfinite(direction[j])) return -INFINITY;
  if (n == 2) {
    warp[2] += direction[0];
    warp[5] += direction[1];
  } else {
    warp[0] += direction[0] / w;
    warp[1] += direction[1] / h;
    warp[2] += direction[2];
    warp[3] += direction[3] / w;
    warp[4] += direction[4] / h;
    warp[5] += direction[5];
  }
  return correlation / sqrt(norm_i * norm_t);
}

inline void accumulate(__private float* totals, __global const float4* pixels,
    int w, int h, int i, __local const float* warp, int n) {
  const int x = i % w, y = i / w;
  // Match OpenCV's fixed-point affine interpolation and nearest-neighbor mask.
  const float2 sx = (float2)(warp[0], warp[3]) * (x * 1024.0f);
  const float2 sy = ((float2)(warp[1], warp[4]) * y + (float2)(warp[2], warp[5])) * 1024.0f;
  const int2 xy = convert_int2_rte(sx) + convert_int2_rte(sy);
  const int2 nearest = (xy + 512) >> 10;
  const int2 grid = (xy + 16) >> 5;
  const int2 p = grid >> 5;
  if (p.x < -1 || p.x >= w || p.y < -1 || p.y >= h) return;
  const float2 f = convert_float2(grid & 31) * (1.0f / 32.0f);
  const float3 row0 = mix(sample(pixels, w, h, p.x, p.y),
                          sample(pixels, w, h, p.x + 1, p.y), f.x);
  const float3 row1 = mix(sample(pixels, w, h, p.x, p.y + 1),
                          sample(pixels, w, h, p.x + 1, p.y + 1), f.x);
  const float3 warped = mix(row0, row1, f.y);
  const bool active = nearest.x >= 0 && nearest.x < w && nearest.y >= 0 && nearest.y < h;
  const float t = pixels[i].x, v = warped.x;
  if (active) {
    totals[0] += 1;
    totals[1] += v;
    totals[2] += t;
    totals[3] += v * v;
    totals[4] += t * t;
    totals[5] += v * t;
  }
  const float jacobian[6] = {n == 2 ? warped.y : warped.y * x / w,
                            n == 2 ? warped.z : warped.y * y / h, warped.y,
                            warped.z * x / w, warped.z * y / h, warped.z};
  for (int j = 0, k = 24; j < n; ++j) {
    const float g = jacobian[j];
    if (active) {
      totals[6 + j] += g;
      totals[18 + j] += g * t;
    }
    totals[12 + j] += g * v;
    for (int a = 0; a <= j; ++a, ++k) totals[k] += g * jacobian[a];
  }
}

__kernel void register_regions(__global const Task* tasks, __global const float4* prepared,
    __global float* result) {
  const int lane = get_local_id(0), region = get_group_id(0);
  const Task t = tasks[region];
  __global const float4* pixels = prepared + t.offset;
  __local float partial[LANES][STATS];
  __local float totals[STATS], warp[6], correlation;
  __local int done;
  if (lane == 0) {
    for (int j = 0; j < 6; ++j) warp[j] = t.warp[j];
    correlation = -1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int iteration = 0; iteration < t.iterations; ++iteration) {
    float thread_totals[STATS] = {0};
    if (t.parameters == 2) {
      for (int i = lane; i < t.width * t.height; i += LANES)
        accumulate(thread_totals, pixels, t.width, t.height, i, warp, 2);
    } else {
      for (int i = lane; i < t.width * t.height; i += LANES)
        accumulate(thread_totals, pixels, t.width, t.height, i, warp, 6);
    }
    for (int j = 0; j < STATS; ++j) partial[lane][j] = thread_totals[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < STATS) {
      float sum = 0;
      for (int j = 0; j < LANES; ++j) sum += partial[j][lane];
      totals[lane] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) {
      const float rho = update(totals, t.parameters, t.width, t.height, warp);
      done = !isfinite(rho) || fabs(rho - correlation) < 1e-5f;
      correlation = rho;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (done) break;
  }
  if (lane == 0) {
    for (int j = 0; j < 6; ++j) result[region * 7 + j] = warp[j];
    result[region * 7 + 6] = correlation;
  }
}
)CLC";

std::size_t roundUp(std::size_t n) {
  return (n + work_group_size - 1) / work_group_size * work_group_size;
}
cv::ocl::Program registrationProgram() {
  static const cv::ocl::ProgramSource program_source("rune", "batched_ecc", source, "");
  cv::String error;
  auto program = cv::ocl::Context::getDefault().getProg(program_source, "", error);
  if (!program.ptr()) throw std::runtime_error("failed to compile rune OpenCL ECC: " + error);
  return program;
}
} // namespace

void prepareOpenCLEcc() { (void)registrationProgram(); }

std::vector<EccResult> estimateOpenCLEcc(const cv::UMat& before, const cv::UMat& after,
                                       std::span<const EccRegion> regions,
                                       EccPreparedBatch& prepared,
                                       std::span<const std::size_t> retry_indices) {
  if (regions.empty()) return {};
  const bool retry = !retry_indices.empty();
  std::vector<Task> tasks;
  tasks.reserve(regions.size());
  int area = 0, largest = 0;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto& r = regions[i];
    const int offset = retry ? prepared.offsets[retry_indices[i]] : area;
    Task t{r.roi.x, r.roi.y, r.roi.width, r.roi.height, offset,
           r.affine ? 6 : 2, r.iterations, {}};
    std::copy_n(r.initial.val, 6, t.warp);
    tasks.push_back(t);
    if (!retry) {
      const int pixels = r.roi.area();
      if (pixels > std::numeric_limits<int>::max() - area)
        throw std::length_error("rune OpenCL registration batch is too large");
      prepared.offsets.push_back(offset);
      area += pixels;
      largest = std::max(largest, pixels);
    }
  }
  const auto program = registrationProgram();
  cv::ocl::Kernel ecc("register_regions", program);
  cv::UMat buffers, blurred;
  cv::UMat output(1, static_cast<int>(regions.size() * 7), CV_32F);
  cv::Mat(1, static_cast<int>(tasks.size() * sizeof(Task)), CV_8U, tasks.data()).copyTo(buffers);
  std::size_t global[]{roundUp(largest), regions.size()}, local[]{work_group_size, 1};
  if (!retry) {
    blurred.create(1, area, CV_32FC2);
    prepared.pixels.create(1, area, CV_32FC4);
    cv::ocl::Kernel blur("blur_regions", program), gradient("prepare_gradients", program);
    if (!blur.args(KernelArg::ReadOnlyNoSize(before), KernelArg::ReadOnlyNoSize(after),
                   KernelArg::PtrReadOnly(buffers), KernelArg::PtrWriteOnly(blurred)).run_(2, global, local, false) ||
        !gradient.args(KernelArg::PtrReadOnly(buffers), KernelArg::PtrReadOnly(blurred),
                       KernelArg::PtrWriteOnly(prepared.pixels)).run_(2, global, local, false))
      throw std::runtime_error("failed to prepare rune OpenCL ECC regions");
  }
  global[0] = work_group_size * regions.size();
  if (!ecc.args(KernelArg::PtrReadOnly(buffers), KernelArg::PtrReadOnly(prepared.pixels),
                KernelArg::PtrWriteOnly(output)).run_(1, global, local, false))
    throw std::runtime_error("failed to dispatch rune OpenCL ECC batch");
  cv::Mat values;
  output.copyTo(values);
  std::vector<EccResult> results(regions.size());
  const float* row = values.ptr<float>();
  for (auto& result : results) {
    std::copy_n(row, 6, result.transform.val);
    result.correlation = row[6];
    row += 7;
  }
  return results;
}
} // namespace Rune::detail
