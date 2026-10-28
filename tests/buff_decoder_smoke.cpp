#include "l2_perception/buff/buff_decoder.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace L2Perception;
using L1Sensor::EnemyColor;

namespace {
void require(bool ok, const char* message)
{
  if (!ok) throw std::runtime_error(message);
}

InferenceResult output(const std::vector<std::array<float, 16>>& rows)
{
  InferenceTensor tensor;
  tensor.name = "output0";
  tensor.shape = {1, 16, rows.size()};
  std::vector<float> values(rows.size() * 16);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t c = 0; c < 16; ++c) values[c * rows.size() + i] = rows[i][c];
  }
  tensor.setOwnedData(std::move(values));
  return {{std::move(tensor)}};
}

std::array<float, 16> candidate(int cls, float x = 320, float score = 0.9F)
{
  std::array<float, 16> row{x, 320, 60, 60};
  row[4 + cls] = score;
  const std::array<cv::Point2f, 4> points{
    cv::Point2f{x, 300}, {x + 20, 320}, {x, 340}, {x - 20, 320}};
  for (std::size_t i = 0; i < 4; ++i) {
    row[8 + i * 2] = points[i].x;
    row[9 + i * 2] = points[i].y;
  }
  return row;
}
}

int main()
{
  try {
    Buff::Decoder decoder;
    const ImageTransform transform{
      .source_size = {1280, 720}, .model_size = {640, 640},
      .source_to_model_scale = 0.5, .pad_top = 140, .pad_bottom = 140};
    const auto mixed = output({candidate(0, 100), candidate(1, 200),
                               candidate(2, 320), candidate(3, 450),
                               candidate(3, 321, 0.7F)});
    const auto red = decoder.decode(mixed, transform, EnemyColor::Red);
    const auto blue = decoder.decode(mixed, transform, EnemyColor::Blue);
    require(red.size() == 2 && blue.size() == 2, "color filtering or cross-state NMS failed");
    require(red[0].class_id == 2 && !red[0].active && red[1].active,
            "RR is a target, RW is already active");
    require(blue[0].class_id == 0 && !blue[0].active && blue[1].active,
            "BR is a target, BW is already active");
    require(cv::norm(red[0].center - cv::Point2f{640, 360}) < 1e-4 &&
            cv::norm(red[0].keypoints[0] - cv::Point2f{640, 320}) < 1e-4,
            "centered letterbox coordinates must invert to original image");
    require(decoder.decode(mixed, transform, EnemyColor::Unknown).empty(),
            "unknown enemy must not select targets");

    auto bad = candidate(2);
    bad[8] = std::numeric_limits<float>::quiet_NaN();
    auto degenerate = candidate(2);
    for (int i = 8; i < 16; ++i) degenerate[i] = 320;
    auto padded = candidate(2);
    padded[9] = 10; // 关键点落在顶部 padding 内。
    require(decoder.decode(output({bad, degenerate, padded, candidate(2, 100, 0.1F)}),
                           transform, EnemyColor::Red).empty(),
            "bad geometry, padding and low confidence must be rejected");
    auto wrong_layout = mixed;
    wrong_layout.outputs[0].shape = {1, 5, 16};
    bool rejected = false;
    try { (void)decoder.decode(wrong_layout, transform, EnemyColor::Red); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "transposed/wrong model must not be silently decoded");
    auto truncated = mixed;
    truncated.outputs[0].setOwnedData({1.0F});
    rejected = false;
    try { (void)decoder.decode(truncated, transform, EnemyColor::Red); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "truncated tensor must be rejected before indexing");
    std::cout << "Buff decoder smoke test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
