#pragma once

#include <opencv2/core.hpp>
#include <span>
#include <vector>

namespace Rune::detail {

struct EccRegion {
  cv::Rect roi;
  cv::Matx23f initial;
  bool affine;
  int iterations;
};

struct EccResult {
  cv::Matx23f transform;
  float correlation;
};

struct EccPreparedBatch {
  cv::UMat pixels;
  std::vector<int> offsets;
};

void prepareOpenCLEcc();
[[nodiscard]] std::vector<EccResult> estimateOpenCLEcc(
    const cv::UMat& before, const cv::UMat& after, std::span<const EccRegion> regions,
    EccPreparedBatch& prepared, std::span<const std::size_t> retry_indices);

} // namespace Rune::detail
