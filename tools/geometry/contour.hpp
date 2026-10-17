#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <optional>
#include <vector>

namespace tools::geometry {

struct ContourDistanceField {
  cv::Rect bounds;
  cv::Mat distances; // CV_32FC1; coordinates are relative to bounds.tl().
};

std::optional<ContourDistanceField> makeContourDistanceField(
    const std::vector<cv::Point>& contour, int padding_px, std::int64_t max_pixels);
// Image pixel coordinates; return cap outside the bilinear sampling domain.
double sampleDistanceField(const ContourDistanceField& field, cv::Point2d pixel, double cap);
// Append uniformly spaced samples along a closed contour.
void appendContourSamples(const std::vector<cv::Point>& contour,
                          std::vector<cv::Point2d>& output, int count = 96);

} // namespace tools::geometry
