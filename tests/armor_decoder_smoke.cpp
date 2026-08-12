#include "l2_perception/armor/armor_decoder.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr std::size_t kFieldCount = 22;

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] float logit(float probability)
{
  return std::log(probability / (1.0F - probability));
}

[[nodiscard]] bool near(float actual, float expected, float tolerance = 1e-4F)
{
  return std::abs(actual - expected) <= tolerance;
}

void setCandidate(std::vector<float>& values, std::size_t candidate,
                  const std::array<cv::Point2f, 4>& raw_corners, float confidence,
                  std::size_t color, std::size_t armor_class, float best_class_score)
{
  const std::size_t row = candidate * kFieldCount;
  for (std::size_t corner = 0; corner < raw_corners.size(); ++corner) {
    values[row + corner * 2] = raw_corners[corner].x;
    values[row + corner * 2 + 1] = raw_corners[corner].y;
  }
  values[row + 8] = logit(confidence);

  for (std::size_t index = 0; index < 4; ++index) {
    values[row + 9 + index] = -5.0F;
  }
  values[row + 9 + color] = 5.0F;

  for (std::size_t index = 0; index < 9; ++index) {
    values[row + 13 + index] = -5.0F;
  }
  values[row + 13 + armor_class] = best_class_score;
}

}  // namespace

int main()
{
  try {
    constexpr std::size_t candidate_count = 4;
    std::vector<float> values(candidate_count * kFieldCount, 0.0F);

    // 模型原始角点顺序为：左上、左下、右下、右上。
    setCandidate(values, 0,
                 {{{100.0F, 100.0F}, {100.0F, 200.0F}, {300.0F, 200.0F}, {300.0F, 100.0F}}}, 0.95F,
                 0, 3, -4.0F);
    // 与候选 0 高度重叠；虽然类别分数更高，但按 objectness 排序时应被抑制。
    setCandidate(values, 1,
                 {{{105.0F, 105.0F}, {105.0F, 205.0F}, {305.0F, 205.0F}, {305.0F, 105.0F}}}, 0.90F,
                 1, 4, 10.0F);
    // 独立的红色英雄装甲。
    setCandidate(values, 2,
                 {{{400.0F, 300.0F}, {400.0F, 350.0F}, {500.0F, 350.0F}, {500.0F, 300.0F}}}, 0.88F,
                 1, 1, 3.0F);
    // 通过 0.7 初筛但没有严格超过 0.8 的最终门限。
    setCandidate(values, 3, {{{20.0F, 20.0F}, {20.0F, 40.0F}, {60.0F, 40.0F}, {60.0F, 20.0F}}},
                 0.75F, 0, 0, 4.0F);

    L2Perception::InferenceTensor tensor;
    tensor.name = "output";
    tensor.shape = {1, candidate_count, kFieldCount};
    tensor.setOwnedData(std::move(values));
    L2Perception::InferenceResult result;
    result.outputs.push_back(std::move(tensor));

    const L2Perception::ImageTransform transform{.source_size = {1280, 960},
                                                 .model_size = {640, 640},
                                                 .source_to_model_scale = 0.5F,
                                                 .pad_left = 10,
                                                 .pad_top = 20,
                                                 .pad_right = 0,
                                                 .pad_bottom = 0};

    const L2Perception::ArmorDecoder decoder;
    const auto detections = decoder.decode(result, transform);
    require(detections.size() == 2, "thresholds/NMS did not retain exactly two candidates");

    const auto& blue = detections[0];
    require(blue.color == L2Perception::ArmorColor::Blue, "color index 0 must map to blue");
    require(blue.class_id == 3, "class argmax was not preserved");
    require(near(blue.confidence, 0.95F), "objectness sigmoid was not preserved");
    require(near(blue.corners[0].x, 180.0F) && near(blue.corners[0].y, 160.0F),
            "left-top point or inverse letterbox transform is wrong");
    require(near(blue.corners[1].x, 580.0F) && near(blue.corners[1].y, 160.0F),
            "model point 3 must become Armor right-top");
    require(near(blue.corners[2].x, 580.0F) && near(blue.corners[2].y, 360.0F),
            "model point 2 must become Armor right-bottom");
    require(near(blue.corners[3].x, 180.0F) && near(blue.corners[3].y, 360.0F),
            "model point 1 must become Armor left-bottom");
    require(near(blue.center.x, 380.0F) && near(blue.center.y, 260.0F),
            "Armor center was not derived from source-image corners");

    const auto& red = detections[1];
    require(red.color == L2Perception::ArmorColor::Red, "color index 1 must map to red");
    require(red.class_id == 1, "second class argmax is wrong");

    bool rejected_invalid_order = false;
    try {
      L2Perception::ArmorDecoderConfig bad_config;
      bad_config.corner_order = {0, 0, 2, 3};
      const L2Perception::ArmorDecoder bad_decoder(bad_config);
      (void)bad_decoder;
    } catch (const std::invalid_argument&) {
      rejected_invalid_order = true;
    }
    require(rejected_invalid_order, "invalid corner-order configuration was accepted");

    std::cout << "armor decoder smoke passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "armor decoder smoke failed: " << error.what() << '\n';
    return 1;
  }
}
